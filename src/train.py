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

def build_optimizer_step(optimizer, initial_params, loss_fun, gradient_clip=None):
  """Builds training step function."""

  #Destructure the optimizer triple.
  init_opt, update_opt, get_params = optimizer
  opt_state = init_opt(initial_params)

  @jax.jit
  def optimizer_step_noclip(current_step, state, batch):
    """Takes a single optimization step."""
    p = get_params(state)
    loss, gradients = jax.value_and_grad(loss_fun)(p, batch)

    new_state = update_opt(current_step, gradients, state)
    return current_step + 1, new_state, loss

  @jax.jit
  def optimizer_step_clip(current_step, state, batch):
    """Takes a single optimization step."""
    p = get_params(state)
    loss, gradients = jax.value_and_grad(loss_fun)(p, batch)

    gradients = optimizers.clip_grads(gradients, gradient_clip)

    new_state = update_opt(current_step, gradients, state)
    return current_step + 1, new_state, loss

  if gradient_clip is None:
    return opt_state, optimizer_step_noclip
  else:
    return opt_state, optimizer_step_clip

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
  if config['data']['dataset'] == 'imdb':
    encoder, train_dset, test_dset = data.imdb(config['data']['max_pad'],
                                               config['data']['batch_size'])
  elif config['data']['dataset'] == 'yelp':
    encoder, train_dset, test_dset = data.yelp(config['data']['max_pad'],
                                               config['data']['batch_size'])

  batch = next(tfds.as_numpy(train_dset))

  # Build network.
  cell = get_cell(config['model']['cell_type'],
                  num_units=config['model']['num_units'])

  init_fun, apply_fun, emb_apply, readout_apply = network.build_rnn(encoder.vocab_size,
                                                                    config['model']['emb_size'],
                                                                    cell,
                                                                    num_outputs=config['model']['num_classes'])

  loss_fun = utils.make_loss_function(apply_fun, loss_type = 'xent',
                                      num_outputs=config['model']['num_classes'])
  acc_fun = utils.make_acc_fun(apply_fun,
                               num_outputs = config['model']['num_classes'])

  _, initial_params = init_fun(prng_key, batch['inputs'].shape)

  logging.info('Initial loss: %0.5f', loss_fun(initial_params, batch))

  # Build training step function.
  learning_rate = optimizers.exponential_decay(config['optim']['base_lr'],
                                               config['optim']['lr_decay_steps'],
                                               config['optim']['lr_decay_rate'])
  opt = optimizers.adam(learning_rate)
  get_params = opt[2]
  opt_state, step_fun = build_optimizer_step(opt, initial_params,
                                             loss_fun,
                                             gradient_clip=config['optim']['gradient_clip'])
  global_step = 0

  # Set up measurement
  def should_report_train(step_num):
    return step_num % config['save']['measure_every'] == 0
  def should_report_test(step_num):
    return step_num % (100*config['save']['measure_every']) == 0

  # Train
  for epoch in range(config['optim']['num_epochs']):

    # Train for one epoch.
    batch_num = 0
    for batch in tfds.as_numpy(train_dset):
      global_step, opt_state, loss = step_fun(global_step, opt_state, batch)
      batch_num = batch_num + 1

      if should_report_train(global_step):
        params = get_params(opt_state)

        # Train accuracy
        accuracy = acc_fun(params, batch)
        logging.info(f'Step {global_step} loss {loss} accuracy {accuracy} (TRAIN)')
        reporter['train'].report_all(global_step, {'loss': loss, 'acc': accuracy})

      if should_report_test(global_step):
        # Test metrics
        is_correct = []
        params = get_params(opt_state)
        for batch in tfds.as_numpy(test_dset):
          is_correct += [acc_fun(params, batch)]
        step_test_acc = np.mean(is_correct)
        reporter['test'].report_all(global_step, {'acc': step_test_acc})
        logging.info(f'Step {global_step} accuracy {step_test_acc} (TEST)')

  final_params = {'params': params}
  reporters.save_dict(config, final_params, 'final_params')

if __name__  == '__main__':
  app.run(main)
