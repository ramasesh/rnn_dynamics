import os
import tensorflow_datasets as tfds
import csv

def readfile(encoder, filename, star_to_label, three_column=False):
  """
  Star to label is a dictionary encoding how to map the
    number of stars to an actual label.
  """
  with open(filename, 'r') as f:
    for row in csv.reader(f):

      if three_column:
        score, _, text = row
      else:
        score, text = row

      if int(score) not in star_to_label.keys():
        continue

      yield {'text': encoder.encode(text),
             'label': star_to_label[int(score)]}
