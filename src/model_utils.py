import jax
import jax.numpy
import pickle

def load_params(filename):
  """ loads a dictionarhy {'params': params} """
  with open(filename, 'rb') as f:
    params = pickle.load(f)

  return params['params']

def combine(parameters_1, parameters_2):
  """ Assumes that we want to use all the parameters_2 except for the final
  (reaodut) layer, for which we want to use parameters_1 """

  return parameters_2[:-1] + (parameters_1[-1],)

def initialize(initial_params, model_config):
  if model_config['pretrained'] is not None:
    preloaded_params = load_params(model_config['pretrained'])
    initial_params = combine(initial_params, preloaded_params)
  return initial_params
