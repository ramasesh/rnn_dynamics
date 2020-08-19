import functools

from absl import app, logging, flags

import json

import jax
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp

import numpy as np

from renn.rnn import unroll, network
import renn
from renn import utils
from src import data, reporters, argparser, model_utils, optim_utils, measurements

FLAGS = flags.FLAGS

def main(_):
  """Builds and trains a sentiment classification RNN."""

  # Get and save config
  config = argparser.parse_args('toy')
  logging.info(json.dumps(config, indent=2))
  reporters.save_config(config)

  prng_key = random.PRNGKey(config['run']['seed'])

  # Load data
  train_dset = renn.data.Unordered(num_classes=config['data']['num_classes'],
                                        batch_size=config['data']['batch_size'],
                                        length_sampler=config['data']['length_sampler'],
                                        sampler_params=config['data']['sampler_params'])
  vocab_size = len(train_dset.vocab)
  max_length = train_dset.max_len

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

  _, initial_params = init_fun(prng_key, (config['data']['batch_size'],
                                          max_length))

  initial_params = model_utils.initialize(initial_params, config['model'])

  # get optimizer
  opt, get_params, opt_state, step_fun = optim_utils.optimization_suite(initial_params,
                                                                        loss_fun,
                                                                        config['optim'])

  # Train
  global_step = 0
  loss = np.nan
  for step in range(config['optim']['steps']):
    batch = next(train_dset)

    dynamic_state = {'opt_state': opt_state,
                     'batch_train_loss': loss,
                     'batch': batch}

    global_step, opt_state, loss = step_fun(global_step, opt_state, batch)

    if global_step % 100 == 1:
      params = get_params(opt_state)
      acc = acc_fun(params, batch).item()
      print(f'Step {global_step}, loss {loss}, acc {acc}')

  final_params = {'params': np.asarray(get_params(opt_state), dtype=object)}
  reporters.save_dict(config, final_params, 'final_params')

if __name__  == '__main__':
  app.run(main)
