"""Data utilities."""

import tensorflow as tf
import tensorflow_datasets as tfds
import csv
from src import data_utils

def imdb(sequence_length, batch_size):
  """Loads the IMDB sentiment dataset.

  Args:
    sequence_length: int, Sequence length for each example.  All examples will
      be padded to this length, and examples longer than this length will get
      truncated to this length (enforces fixed length sequences).
      TODO is sequence_length in units of words, subword tokens, what?
    batch_size: int, Number of examples to group in a minibatch.
    bufsize: int, Size of the shuffle bufer (Default: 1024).
    shuffle_seed: int, Random seed for hte shuffle operation (Default: 0).

  Returns:
    encoder: a TensorFlow Datasets Text Encoder object.
    info: a TensorFlow Datasets info object.
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

def yelp(sequence_length, batch_size):

  vocab_filename = './data/vocab/yelp'
  encoder = data_utils.get_encoder(vocab_filename)

  dset_types = ['train', 'test']

  filenames = {'test': './data/yelp/test.csv',
               'train': './data/yelp/train.csv'}

  def train_iterator():
    return data_utils.readfile(encoder, filenames['train'])
  def test_iterator():
    return data_utils.readfile(encoder, filenames['test'])

  output_types = {'text': tf.int64, 'label': tf.int64}

  datasets = {'train': tf.data.Dataset.from_generator(train_iterator, output_types),
              'test': tf.data.Dataset.from_generator(test_iterator, output_types)}

  train_dset = pipeline(datasets['train'], sequence_length, batch_size)
  test_dset = pipeline(datasets['test'], sequence_length, batch_size)

  return encoder, train_dset, test_dset

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
