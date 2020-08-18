import functools

from absl import app, logging, flags

import json

import jax
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp

import numpy as np

from renn.rnn import unroll, network
from renn import utils
from src import data, reporters, argparser, model_utils, optim_utils, measurements, manager, toy_data

FLAGS = flags.FLAGS

def main(_):
  """Builds and trains a sentiment classification RNN."""

  # Get and save config
  config = argparser.parse_args()
  logging.info(json.dumps(config, indent=2))
  reporters.save_config(config)

  prng_key = random.PRNGKey(config['run']['seed'])

  # Load data
  vocab_size = len(toy_data.numerical_vocab)
  train_dset = toy_data.ToyDataset(40, 20)

  # Build network.
  cell = model_utils.get_cell(config['model']['cell_type'],
                              num_units=config['model']['num_units'])

  init_fun, apply_fun, _, _ = network.build_rnn(vocab_size,
                                                config['model']['emb_size'],
                                                cell,
                                                config['model']['num_outputs'])

  loss_fun, acc_fun = optim_utils.loss_and_accuracy(apply_fun,
                                                    config['model'],
                                                    config['optim'])

  BATCH_SIZE=64
  MAX_LENGTH=100
  _, initial_params = init_fun(prng_key, (BATCH_SIZE, MAX_LENGTH))

  initial_params = model_utils.initialize(initial_params, config['model'])

  # get optimizer
  opt, get_params, opt_state, step_fun = optim_utils.optimization_suite(initial_params,
                                                                        loss_fun,
                                                                        config['optim'])

  # Train
  num_steps = 10000
  global_step = 0
  loss = np.nan
  for step in range(num_steps):
    batch = next(train_dset)

    dynamic_state = {'opt_state': opt_state,
                     'batch_train_loss': loss,
                     'batch': batch}

    global_step, opt_state, loss = step_fun(global_step, opt_state, batch)

    if global_step % 100 == 0:
      params = get_params(opt_state)
      acc = acc_fun(params, batch).item()
      print(f'Step {global_step}, loss {loss}, acc {acc}')

  final_params = {'params': np.asarray(get_params(opt_state), dtype=object)}
  reporters.save_dict(config, final_params, 'final_params')

if __name__  == '__main__':
  app.run(main)
