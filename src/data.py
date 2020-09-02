"""Data utilities."""

import tensorflow as tf
import tensorflow_datasets as tfds
import csv
from src import data_utils
from renn.data import datasets

# Mapping from 5-class categories to subclasses
SENTIMENT_CLASS_MAPPINGS = {1: {4:1, 5:1, 1:0, 2:0},
                  2: {4:1, 5:1, 1:0, 2:0},
                  3: {1: 0, 3: 1, 5: 2},
                  5: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}}

def get_dataset(data_config):
  if data_config['dataset'] == 'imdb':
    # currently, we copy the vocab file from renn
    # before job submission.  Is there a way to make sure these are synced?
    vocab_file = './data/vocab/imdb.vocab'
    vocab_size = len(open(vocab_file, 'r').readlines())

    train_dset = datasets.imdb('train', vocab_file, data_config['max_pad'],
                               data_config['batch_size'])
    test_dset = datasets.imdb('test', vocab_file, data_config['max_pad'],
                               data_config['batch_size'])

  elif data_config['dataset'] == 'yelp':
    encoder, train_dset, test_dset = yelp(data_config['max_pad'],
                                          data_config['batch_size'],
                                          data_config['num_classes'])
    vocab_size = encoder.vocab_size

  elif data_config['dataset'] == 'dbpedia':
    encoder, train_dset, test_dset = dbpedia(data_config['max_pad'],
                                          data_config['batch_size'],
                                          data_config['num_classes'])
    vocab_size = encoder.vocab_size

  elif data_config['dataset'] == 'ag_news':
    # currently, we copy the vocab file from renn
    # before job submission.  Is there a way to make sure these are synced?
    vocab_file = './data/vocab/ag_news.vocab'
    vocab_size = len(open(vocab_file, 'r').readlines())

    def filter_fn(item):
      return item['labels'] < data_config['num_classes']

    train_dset = datasets.ag_news('train', vocab_file, data_config['max_pad'],
                               data_config['batch_size'], filter_fn=filter_fn,
                               data_dir='./data')
    test_dset = datasets.ag_news('test', vocab_file, data_config['max_pad'],
                               data_config['batch_size'], filter_fn=filter_fn,
                               data_dir='./data')

  return vocab_size, train_dset, test_dset

def dbpedia(sequence_length, batch_size, num_classes=14):
  """
  Returns the DBPedia dataset, with the specified number of classes.

  Arguments:
    sequence_length -
    batch_size -
    num_classes - the number of classes to divide up the dataset into.
                  Allowed number of classes are 2, 3, ..., 14
                  We keep classes 0, 1, ..., num_classes - 1
  """

  class_label = {x: x - 1 for x in range(1,num_classes+1)}
  print(class_label)

  vocab_filename = './data/vocab/yelp'
  encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)

  dset_types = ['train', 'test']
  output_types = {'text': tf.int64,
                  'label': tf.int64}

  train_filename = './data/dbpedia/train.csv'
  train_iterator = lambda : data_utils.readfile(encoder, train_filename,
                                          class_label, three_column=True)
  train_dataset = tf.data.Dataset.from_generator(train_iterator, output_types)
  pipelined_train = pipeline(train_dataset, sequence_length, batch_size, bufsize=262144)

  test_filename = './data/dbpedia/test.csv'
  test_iterator = lambda : data_utils.readfile(encoder, test_filename,
                                          class_label, three_column=True)
  test_dataset = tf.data.Dataset.from_generator(test_iterator, output_types)
  pipelined_test = pipeline(test_dataset, sequence_length, batch_size)

  return encoder, pipelined_train, pipelined_test

def yelp(sequence_length, batch_size, num_classes=5):
  """
  Returns the Yelp dataset, with a specified number of classes.

  Arguments:
    sequence_length -
    batch_size -
    num_classes - the number of classes to divide up the dataset into.
                  Allowed number of classes are {1, 2, 3, and 5}
                  NOTE that 1 is a bit of a misnomer, and we don't actually
                  give a single class in this case.  Here's what we do:
                  num_classes == 1 or 2:
                    star-classes 4 and 5 are mapped to 1
                    star-classes 1 and 2 are mapped to 0
                    star-class 3 is ignored
                  num_classes == 3:
                    star-class 1 is mapped to 0
                    star-class 3 is mapped to 1
                    star-class 5 is mapped to 2
                    star-classes 2 and 4 are ignored
                  num_classes == 5:
                    all classes are given with their proper labels
  """

  star_to_label = SENTIMENT_CLASS_MAPPINGS[num_classes]

  vocab_filename = './data/vocab/yelp'
  encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)

  dset_types = ['train', 'test']
  output_types = {'text': tf.int64,
                  'label': tf.int64}

  train_filename = './data/yelp/train.csv'
  train_iterator = lambda : data_utils.readfile(encoder, train_filename,
                                          star_to_label, three_column=False)
  train_dataset = tf.data.Dataset.from_generator(train_iterator, output_types)
  pipelined_train = pipeline(train_dataset, sequence_length, batch_size)

  test_filename = './data/yelp/test.csv'
  test_iterator = lambda : data_utils.readfile(encoder, test_filename,
                                          star_to_label, three_column=False)
  test_dataset = tf.data.Dataset.from_generator(test_iterator, output_types)
  pipelined_test = pipeline(test_dataset, sequence_length, batch_size)

  return encoder, pipelined_train, pipelined_test

def pipeline(dset, sequence_length, batch_size, bufsize=1024, shuffle_seed=0):
  """Data preprocessing pipeline."""

  # Truncates examples longer than the sequence length.
  dset = dset.filter(lambda d: len(d['text']) <= sequence_length)

  def _extract(d):
    return {
        'inputs': d['text'],
        'labels': d['label'],
        'index': len(d['text'])
    }
  dset = dset.map(_extract)

  # Cache, shuffle, and pad.
  dset = dset.cache().shuffle(buffer_size=bufsize, seed=shuffle_seed)

  # Pad
  padded_shapes = {
      'inputs': (sequence_length,),
      'labels': (),
      'index': (),
  }
  dset = dset.padded_batch(batch_size, padded_shapes)

  return dset

