"""Data utilities."""

import tensorflow_datasets as tfds

def imdb(sequence_length, batch_size, config='subwords8k',
         bufsize=1024, shuffle_seed=0):
  """Loads the IMDB sentiment dataset.

  Args:
    sequence_length: int, Sequence length for each example.  All examples will
      be padded to this length, and examples longer than this length will get
      truncated to this length (enforces fixed length sequences).
      TODO is sequence_length in units of words, subword tokens, what?
    batch_size: int, Number of examples to group in a minibatch.
    config: str, Specifies which configuration to load (Default: subwords8k).
    bufsize: int, Size of the shuffle bufer (Default: 1024).
    shuffle_seed: int, Random seed for hte shuffle operation (Default: 0).

  Returns:
    encoder: a TensorFlow Datasets Text Encoder object.
    info: a TensorFlow Datasets info object.
    train_dset: a TensorFlow Dataset for training.
    test_dset: a TensorFlow Dataset for testing.
  """
  dset_name = f'imdb_reviews/{config}'

  import os
  print("Contents of data")
  print(os.listdir('./data'))

  # Load raw datasets.
  datasets, info = tfds.load(dset_name, with_info=True, download=False, data_dir='./data/')
  encoder = info.features['text'].encoder

  def pipeline(dset):
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

  train_dset = pipeline(datasets['train'])
  test_dset = pipeline(datasets['test'])

  return encoder, info, train_dset, test_dset
