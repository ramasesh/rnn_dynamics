"""Data utilities."""

import tensorflow as tf
import tensorflow_datasets as tfds
import csv
from src import data_utils

# Mapping from 5-class categories to subclasses
CLASS_MAPPINGS = {1: {4:1, 5:1, 1:0, 2:0},
                  2: {4:1, 5:1, 1:0, 2:0},
                  3: {1: 0, 3: 1, 5: 2},
                  5: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}}

def get_dataset(data_config):
  if data_config['dataset'] == 'imdb':
    encoder, train_dset, test_dset = imdb(data_config['max_pad'],
                                          data_config['batch_size'])
  elif data_config['dataset'] == 'yelp':
    encoder, train_dset, test_dset = yelp(data_config['max_pad'],
                                          data_config['batch_size'],
                                          data_config['num_classes'])

  return encoder, train_dset, test_dset

def imdb(sequence_length, batch_size):
  """Loads the IMDB sentiment dataset.

  Args:
    sequence_length: int, Sequence length for each example.  All examples will
      be padded to this length, and examples longer than this length will get
      truncated to this length (enforces fixed length sequences).
    batch_size: int, Number of examples to group in a minibatch.

  Returns:
    encoder: a TensorFlow Datasets Text Encoder object.
    train_dset: a TensorFlow Dataset for training.
    test_dset: a TensorFlow Dataset for testing.
  """
  config='subwords8k'
  dset_name = f'imdb_reviews/{config}'

  # Load raw datasets.
  datasets, info = tfds.load(dset_name, with_info=True, download=False, data_dir='./data/')
  encoder = info.features['text'].encoder

  train_dset = pipeline(datasets['train'], sequence_length, batch_size)
  test_dset = pipeline(datasets['test'], sequence_length, batch_size)

  return encoder, train_dset, test_dset

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

  star_to_label = CLASS_MAPPINGS[num_classes]

  vocab_filename = './data/vocab/yelp'
  encoder = data_utils.get_encoder(vocab_filename)

  dset_types = ['train', 'test']
  output_types = {'text': tf.int64,
                  'label': tf.int64}

  datasets = {}
  for dset_type in dset_types:
    filename = f'./data/yelp/{dset_type}.csv'
    iterator = lambda : data_utils.readfile(encoder, filename, star_to_label)
    dataset = tf.data.Dataset.from_generator(iterator, output_types)
    datasets[dset_type] = pipeline(dataset, sequence_length, batch_size)

  return encoder, datasets['train'], datasets['test']

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

