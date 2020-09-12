"""Data utilities."""

import tensorflow as tf
import tensorflow_datasets as tfds
import csv
from src import data_utils
from renn.data import datasets


def get_dataset(data_config):
  if data_config['dataset'] == 'imdb':
    vocab_file = './data/vocab/imdb.vocab'
    vocab_size = len(open(vocab_file, 'r').readlines())

    train_dset = datasets.imdb('train', vocab_file, data_config['max_pad'],
                               data_config['batch_size'])
    test_dset = datasets.imdb('test', vocab_file, data_config['max_pad'],
                               data_config['batch_size'])

  elif data_config['dataset'] == 'yelp':
    vocab_file = './data/vocab/yelp.vocab'
    vocab_size = len(open(vocab_file, 'r').readlines())

    train_dset = datasets.yelp('train', data_config['num_classes'],
                               vocab_file, data_config['max_pad'],
                               data_config['batch_size'],
                               data_dir='./data/yelp/')
    test_dset = datasets.yelp('test', data_config['num_classes'],
                               vocab_file, data_config['max_pad'],
                               data_config['batch_size'],
                               data_dir='./data/yelp/')

  elif data_config['dataset'] == 'amazon':
    vocab_file = './data/vocab/amazon.vocab'
    vocab_size = len(open(vocab_file, 'r').readlines())

    train_dset = datasets.amazon('train', data_config['num_classes'],
                               vocab_file, data_config['max_pad'],
                               data_config['batch_size'],
                               data_dir='./data/amazon/')
    test_dset = datasets.amazon('test', data_config['num_classes'],
                               vocab_file, data_config['max_pad'],
                               data_config['batch_size'],
                               data_dir='./data/amazon/')

  elif data_config['dataset'] == 'dbpedia':
    vocab_file = './data/vocab/dbpedia.vocab'
    vocab_size = len(open(vocab_file, 'r').readlines())

    train_dset = datasets.dbpedia('train', data_config['num_classes'],
                               vocab_file, data_config['max_pad'],
                               data_config['batch_size'],
                               data_dir='./data/dbpedia/')
    test_dset = datasets.dbpedia('test', data_config['num_classes'],
                               vocab_file, data_config['max_pad'],
                               data_config['batch_size'],
                               data_dir='./data/dbpedia/')

  elif data_config['dataset'] == 'ag_news':
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

