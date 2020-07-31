""" Handles argument parsing """
import random
import string

from absl import app
from absl import flags
from absl import logging

# for version checks
import jax
import tensorflow_datasets as tfds
import tensorflow

from collections import OrderedDict

# model config
flags.DEFINE_enum('cell_type', 'VanillaRNN', ['VanillaRNN', 'GRU', 'LSTM', 'UGRNN'], 'RNN Cell Type')
flags.DEFINE_integer('emb_size', 128, 'Token embedding dimension')
flags.DEFINE_integer('num_units', 256, 'Hidden state dimension')
flags.DEFINE_string('pretrained', None, 'Filepath of pretrained model parameters to load')

# data config
flags.DEFINE_integer('batch_size', 64, 'Optimization batch size')
flags.DEFINE_integer('max_pad', 1500, 'Max sequence length')
flags.DEFINE_enum('dataset', 'imdb', ['imdb', 'yelp'], 'dataset')
flags.DEFINE_integer('num_classes', None, 'Number of classes in the dataset')

# optim config
flags.DEFINE_float('base_lr', 1e-3, 'Initial learning rate')
flags.DEFINE_float('lr_decay_steps', 1000., 'LR Decay steps')
flags.DEFINE_float('lr_decay_rate', 0.2, 'LR Decay rate')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs to train for')
flags.DEFINE_float('gradient_clip', None, 'Norm of gradient clip')
flags.DEFINE_float('L2', 0.0, 'Coefficient of L2')

# run config
flags.DEFINE_integer('seed', 0, 'Random seed for JAX rngs')

# save config
flags.DEFINE_string('save_location', '.', 'Save location')
flags.DEFINE_string('job_name_template', 'dynamics', 'Job name template')
flags.DEFINE_integer('measure_train', 1000, 'Number of steps between measurements')
flags.DEFINE_integer('measure_test', 1000, 'Number of steps between test measurements.')
flags.DEFINE_integer('checkpoint_interval', 1000, 'Number of steps between checkpointng')

FLAGS = flags.FLAGS

ARGS_BY_TYPE = {'model': ['cell_type', 'emb_size', 'num_units', 'pretrained'],
                'data': ['dataset', 'batch_size', 'max_pad', 'num_classes'],
                'optim': ['base_lr', 'lr_decay_steps', 'lr_decay_rate',
                          'num_epochs', 'gradient_clip', 'L2'],
                'run': ['seed'],
                'save': ['save_location', 'job_name_template', 'measure_train',
                         'measure_test', 'checkpoint_interval']}


def parse_args():
  config = {}

  for config_type in ARGS_BY_TYPE.keys():
    config[f'{config_type}'] = get_config(config_type)
  config['env_info'] = _get_env_info()

  config = populate_save_location(config)
  config = populate_classes_and_outputs(config)

  return config

def populate_classes_and_outputs(config):
  class_options = {'imdb': [2], 'yelp': [2,3,5]}
  class_defaults = {'imdb': 2, 'yelp': 5}

  dataset = config['data']['dataset']

  # convert between number of classes and number of model outputs
  # basically if number of classes is 2, we can get away with one model output
  def classes_to_outputs(x):
    if x == 2:
      return 1
    else:
      return x

  if FLAGS.num_classes is not None:
    assert FLAGS.num_classes in class_options[dataset]
    config['data']['num_classes'] = FLAGS.num_classes
  else:
    config['data']['num_classes'] = class_defaults[dataset]
  config['model']['num_outputs'] = classes_to_outputs(config['data']['num_classes'])
  return config

def get_config(config_type):
  config = {}
  args = ARGS_BY_TYPE[config_type]

  for arg in args:
    config[arg] = getattr(FLAGS, arg)

  return config

def _get_env_info():
  info = OrderedDict({'jax_version': jax.__version__,
                      'tfds_version': tfds.__version__,
                      'tensorflow_version': tensorflow.__version__})
  return info

def _get_random_key(key_len=8):
  char_list = string.ascii_lowercase + string.digits
  return ''.join(random.choices(char_list, k=key_len))

def populate_save_location(config):
  all_arguments = {}
  for key in config.keys():
    all_arguments.update(config[key])

  run_id = _get_random_key()

  job_name = config['save']['job_name_template'].format(**all_arguments)
  job_name = job_name + f'_run_{run_id}'
  config['save'].update({'job_name': job_name})

  return config

