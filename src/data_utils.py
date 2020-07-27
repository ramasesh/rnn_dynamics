import os
import tensorflow_datasets as tfds
import csv

def get_encoder(vocab_file):
  """Given the location of a text_file which contins the words
  comprising a vocabulary, returns an encoder which maps
  text corpora to a list of indices in the vocab"""

  return tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)

def readfile(encoder, filename):
  with open(filename, 'r') as f:
    for row in csv.reader(f):
      score, text = row
      yield {'text': encoder.encode(text),
             'label': int(score)}
