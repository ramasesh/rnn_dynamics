import functools

from absl import app
from absl import logging
from absl import flags

import json

import jax
from jax import random
from jax.experimental import optimizers
from jax.experimental import stax
import jax.numpy as jnp

import numpy as np
import tensorflow_datasets as tfds

from renn.rnn import cells, unroll, network
from renn import utils
from src import data, reporters, argparser

FLAGS = flags.FLAGS

def build_optimizer_step(optimizer, initial_params, loss_fun):
  """Builds training step function."""

  #Destructure the optimizer triple.
  init_opt, update_opt, get_params = optimizer
  opt_state = init_opt(initial_params)

  @jax.jit
  def optimizer_step(current_step, state, batch):
    """Takes a single optimization step."""
    p = get_params(state)
    loss, gradients = jax.value_and_grad(loss_fun)(p, batch)
    new_state = update_opt(current_step, gradients, state)
    return current_step + 1, new_state, loss

  return opt_state, optimizer_step

def get_cell(cell_type, **kwargs):
  """ Builds a cell given the type and passes along any kwargs """

  cell_functions = {'LSTM': cells.LSTM,
                    'GRU':  cells.GRU,
                    'VanillaRNN': cells.VanillaRNN,
                    #'UGRNN': cells.UGRNN
                    }

  if cell_type not in cell_functions.keys():
    raise Exception(f'Input argument cell_type must be in {cell_functions.keys()}')

  return cell_functions[cell_type](**kwargs)

def main(_):
  """Builds and trains a sentiment classification RNN."""

  # Get and save config
  config = argparser.parse_args()
  logging.info(json.dumps(config, indent=2))
  reporters.save_config(config)

  # Set up reporters
  data_store = {}
  reporter = reporters.build_reporters(config['save'],
                                       data_store)
  prefixes = ['test', 'train']
  reporter = reporters.prefix_reporters(reporter,
                                        prefixes)


  prng_key = random.PRNGKey(config['run']['seed'])

  # Load data.
  encoder, _, train_dset, test_dset = data.imdb(config['data']['max_pad'],
                                                config['data']['batch_size'])
  batch = next(tfds.as_numpy(train_dset))

  # Build network.
  cell = get_cell(config['model']['cell_type'],
                  num_units=config['model']['num_units'])

  init_fun, apply_fun, emb_apply, readout_apply = network.build_rnn(encoder.vocab_size,
                                                                    config['model']['emb_size'],
                                                                    cell,
                                                                    num_outputs=1)

  loss_fun = utils.make_loss_function(apply_fun, loss_type = 'xent', num_outputs=1)
  acc_fun = utils.make_acc_fun(apply_fun, num_outputs = 1)

  _, initial_params = init_fun(prng_key, batch['inputs'].shape)

  logging.info('Initial loss: %0.5f', loss_fun(initial_params, batch))

  # Build training step function.
  learning_rate = optimizers.exponential_decay(config['optim']['base_lr'],
                                               config['optim']['lr_decay_steps'],
                                               config['optim']['lr_decay_rate'])
  opt = optimizers.adam(learning_rate)
  get_params = opt[2]
  opt_state, step_fun = build_optimizer_step(opt, initial_params, loss_fun)
  global_step = 0

  # Train.
  train_losses = []
  step_num = 0
  for epoch in range(config['optim']['num_epochs']):
    # Train for one epoch.
    batch_num = 0
    for batch in tfds.as_numpy(train_dset):
      global_step, opt_state, loss = step_fun(global_step, opt_state, batch)
      train_losses.append(loss)
      # logging.info(f'Epoch {epoch+1} Batch {batch_num} Loss: {loss}')
      batch_num = batch_num + 1
      step_num = step_num + 1
      reporter['train'].report_all(step_num, {'loss': loss})

    # Test metrics.
    params = get_params(opt_state)
    is_correct = []
    for batch in tfds.as_numpy(test_dset):
      is_correct += [acc_fun(params, batch)]
    # logging.info('[Epoch %d] Test accuracy: %0.3f',
    #             epoch + 1,
    #             100 * np.mean(is_correct))
    epoch_test_acc = np.mean(is_correct)
    reporter['test'].report_all(step_num, {'acc': epoch_test_acc})

  final_params = {'params': params}
  reporters.save_dict(config, final_params, 'final_params')
  

if __name__  == '__main__':
  app.run(main)
