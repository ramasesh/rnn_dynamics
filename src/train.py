import functools

from absl import app, logging, flags

import json

import jax
from jax import random
from jax.experimental import optimizers
import jax.numpy as jnp

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from renn.rnn import unroll, network
from renn import utils
from src import data, reporters, argparser, model_utils, optim_utils, measurements

import uv
import uv.manager as m
from uv.mlflow.reporter import MLFlowReporter

FLAGS = flags.FLAGS

def main(_):
  with uv.start_run(), uv.active_reporter(MLFlowReporter()):
    """Builds and trains a sentiment classification RNN."""

    # prevent tf from accessing GPU
    tf.config.experimental.set_visible_devices([], "GPU")

    # Get and save config
    config = argparser.parse_args('main')
    logging.info(json.dumps(config, indent=2))

    reporters.save_config(config)

    uv.report_params(reporters.flatten(config))

    prng_key = random.PRNGKey(config['run']['seed'])

    # Load data.
    vocab_size, train_dset, test_dset = data.get_dataset(config['data'])

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
                                            config['data']['max_pad']))

    initial_params = model_utils.initialize(initial_params, config['model'])

    # get optimizer
    opt, get_params, opt_state, step_fun = optim_utils.optimization_suite(initial_params,
                                                                          loss_fun,
                                                                          config['optim'])

    ## Scope setup
    # Reporter setup
    data_store = {}
    reporter = reporters.build_reporters(config['save'],
                                         data_store)
    # Static state for scope
    static_state = {'acc_fun': acc_fun,
                    'loss_fun': loss_fun,
                    'param_extractor': get_params,
                    'test_set': test_dset}

    oscilloscope = m.MeasurementManager(static_state,
                                            reporter)

    oscilloscope.add_measurement({'name': 'test_acc',
                                  'interval': config['save']['measure_test'],
                                  'function': measurements.measure_test_acc})
    oscilloscope.add_measurement({'name': 'shuffled_test_acc',
                                  'interval': config['save']['measure_test'],
                                  'function': measurements.measure_shuffled_acc})
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
    loss = np.nan
    for epoch in range(config['optim']['num_epochs']):

      for batch_num, batch in enumerate(tfds.as_numpy(train_dset)):
        dynamic_state = {'opt_state': opt_state,
                         'batch_train_loss': loss,
                         'batch': batch}

        oscilloscope.measure(int(global_step), dynamic_state)

        global_step, opt_state, loss = step_fun(global_step, opt_state, batch)

        if global_step % config['save']['checkpoint_interval'] == 0:
          params = get_params(opt_state)
          np_params = np.asarray(params, dtype=object)
          reporters.save_dict(config, np_params, f'checkpoint_{global_step}')

    oscilloscope.measure(int(global_step), dynamic_state, measurement_list=['test_acc', 'shuffled_test_acc'])

    final_params = {'params': np.asarray(get_params(opt_state), dtype=object)}
    reporters.save_dict(config, final_params, 'final_params')

if __name__  == '__main__':
  app.run(main)
