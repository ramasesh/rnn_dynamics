import pickle

from renn.rnn import cells

def load_params(filename):
  """ loads a dictionary containign model parameters,
  under the key 'params', and returns the parameters """
  with open(filename, 'rb') as f:
    params = pickle.load(f)

  return params['params']

def combine(parameters_1, parameters_2):
  """ Combines the readout parameter of parameters_1 with all
  other parameters from parameters_2.
  Used for loading from pre-trained parameters"""

  return parameters_2[:-1] + (parameters_1[-1],)

def initialize(initial_params, model_config):
  """ Initializes the parameters of a model.
  If we want to start from a pre-trained model, i.e. if
  model_config['pretrained'] points to a file to load parameters
  from, we load those parameters, and combine those with the
  readout parameters from initial_params.

  Otherwise, we just load the initial_params """

  if model_config['pretrained'] is not None:
    preloaded_params = load_params(model_config['pretrained'])
    initial_params = combine(initial_params, preloaded_params)
  return initial_params

def get_cell(cell_type, **kwargs):
  """ Builds a cell given the type and passes along any kwargs """

  cell_functions = {'LSTM': cells.LSTM,
                    'GRU':  cells.GRU,
                    'VanillaRNN': cells.VanillaRNN,
                    'UGRNN': cells.UGRNN
                    }

  if cell_type not in cell_functions.keys():
    raise Exception(f'Input argument cell_type must be in {cell_functions.keys()}')

  return cell_functions[cell_type](**kwargs)

