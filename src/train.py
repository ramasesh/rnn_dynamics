import functools

from absl import app
from absl import logging
from absl import flags

import json

import jax
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp

import numpy as np
import tensorflow_datasets as tfds

from renn.rnn import unroll, network
from renn import utils
from src import data, reporters, argparser, model_utils, optim_utils, measurements, manager

FLAGS = flags.FLAGS

def main(_):
  """Builds and trains a sentiment classification RNN."""

  # Get and save config
  config = argparser.parse_args()
  logging.info(json.dumps(config, indent=2))
  reporters.save_config(config)

  prng_key = random.PRNGKey(config['run']['seed'])

  # Load data.
  encoder, train_dset, test_dset = data.get_dataset(config['data'])
  batch = next(tfds.as_numpy(train_dset))

  # Build network.
  cell = model_utils.get_cell(config['model']['cell_type'],
                              num_units=config['model']['num_units'])

  init_fun, apply_fun, emb_apply, readout_apply = network.build_rnn(encoder.vocab_size,
                                                                    config['model']['emb_size'],
                                                                    cell,
                                                                    num_outputs=config['model']['num_outputs'])

  loss_fun, acc_fun = optim_utils.loss_and_accuracy(apply_fun,
                                                    config['model'],
                                                    config['optim'])


  _, initial_params = init_fun(prng_key, batch['inputs'].shape)
  initial_params = model_utils.initialize(initial_params, config['model'])

  logging.info('Initial loss: %0.5f', loss_fun(initial_params, batch))

  # get optimizer
  opt, get_params, opt_state, step_fun = optim_utils.optimization_suite(initial_params,
                                                                        loss_fun,
                                                                        config['optim'])

  ## Scope setup
  # Reporter setup
  data_store = {}
  reporter = reporters.build_reporters(config['save'],
                                       data_store)
  # Static state for Multimeter
  static_state = {'acc_fun': acc_fun,
                  'loss_fun': loss_fun,
                  'param_extractor': get_params,
                  'test_set': test_dset}

  oscilloscope = manager.MeasurementManager(static_state,
                                          reporter)

  oscilloscope.add_measurement({'name': 'test_acc',
                                'interval': config['save']['measure_test'],
                                'function': measurements.measure_test_acc})
  oscilloscope.add_measurement({'name': 'train_acc',
                                'interval': config['save']['measure_train'],
                                'function': measurements.measure_batch_acc})
  oscilloscope.add_measurement({'name': 'train_loss',
                                'interval': config['save']['measure_train'],
                                'function': measurements.measure_batch_loss})
  oscilloscope.add_measurement({'name': 'l2_norm',
                                'interval': config['save']['measure_test'],
                                'function': measurements.measure_l2_norm})

  # Train
  global_step = 0

  for epoch in range(config['optim']['num_epochs']):

    for batch_num, batch in enumerate(tfds.as_numpy(train_dset)):
      global_step, opt_state, loss = step_fun(global_step, opt_state, batch)

      dynamic_state = {'opt_state': opt_state,
                       'batch_train_loss': loss,
                       'batch': batch}

      oscilloscope.process(global_step, dynamic_state)
      if global_step % config['save']['checkpoint_interval'] == 0:
        params = get_params(opt_state)
        np_params = np.asarray(params, dtype=object)
        reporters.save_dict(config, np_params, f'checkpoint_{global_step}')

  oscilloscope.trigger_subset(global_step, dynamic_state, ['test_acc'])

  final_params = {'params': np.asarray(get_params(opt_state), dtype=object)}
  reporters.save_dict(config, final_params, 'final_params')

if __name__  == '__main__':
  app.run(main)
