import tensorflow_datasets as tfds
from jax.experimental import optimizers

class AverageMeter:
  """Keeps a running average, used for, e.g., calculating
  accuracy or loss averaged over a training or test set """

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, num):
    self.val = val
    self.sum += val * num
    self.count += num
    self.avg = self.sum / self.count

def measure_batch_acc(state):
  """Measures the accuracy averaged over a given batch"""
  params = get_params(state)
  batch = state['batch']

  return state['acc_fun'](params, batch)

def measure_batch_loss(state):
  """Measures the loss averaged over a given batch"""
  return state['batch_train_loss']

def measure_test_acc(state):
  """Measures the accuracy averaged over the test set"""
  test_set = state['test_set']
  test_acc = AverageMeter()

  params = get_params(state)
  acc_fun = state['acc_fun']

  for batch in tfds.as_numpy(test_set):
    batch_avg = acc_fun(params, batch).item()
    test_acc.update(batch_avg, len(batch['inputs']))

  return test_acc.avg

def measure_l2_norm(state):
  """Measures the l2 norm (NOT squared l2 norm!) of the
  RNN parameters"""
  params = get_params(state)
  embed_params, rnn_params, output_params = params
  return optimizers.l2_norm(rnn_params)

def get_params(state):
  """Returns model parameters from the full state"""
  params = state['param_extractor'](state['opt_state'])
  return params
