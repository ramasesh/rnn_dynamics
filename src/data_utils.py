import numpy as np
import os
import tensorflow_datasets as tfds
import csv

def readfile(encoder, filename, label_converter, three_column=False):
  """
  label_converter is a dictionary encoding how to map the score
  in the csv file to an actual label.
  """
  with open(filename, 'r') as f:
    for row in csv.reader(f):

      if three_column:
        score, _, text = row
      else:
        score, text = row

      if int(score) not in label_converter.keys():
        continue

      yield {'text': encoder.encode(text),
             'label': label_converter[int(score)]}

def shuffle_words(batch):
  """ Returns a batch in which each example is a shuffled
  version of the sentences in the argument batch. """

  batch.update({'inputs': shuffle(batch['inputs'], batch['index'])})
  return batch

def shuffle(sentences, lengths):
  """ Shuffles sentences, respecting the length of the
  sentences """
  n_sentences, sentence_length = sentences.shape
  permuted_indices = np.full(sentences.shape,
                             np.arange(sentence_length))

  for i in range(n_sentences):
    permutation = np.random.permutation(lengths[i])
    permuted_indices[i, :len(permutation)] = permutation

  return reorder(sentences, permuted_indices)

def reorder(input_array, indices):
  """ Reorders row of input_array according to the specification
  in indices

  Arguments:
    input_array: a 2D np array
    indices: a np array with the same shape as input_array

  Returns:
    reordered_array: a 2D np array whose rows have been
                     reordered from input_array

  Example:
    input_array: np.array([['A', 'B', 'C'],
                           ['D', 'E', 'F']])
    indices: np.array([[0,1,2],
                       [2,1,0]])
    reordered_array: np.array([['A', 'B', 'C'],
                               ['F', 'E', 'D']])
  """
  reordered_array = np.array([row[order] for row, order in zip(input_array, indices)])
  return reordered_array
