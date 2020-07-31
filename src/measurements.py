class AverageMeter:
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
  params = state['param_extractor'](state['opt_state'])
  batch = state['batch']

  return state['acc_fun'](params, batch)

def measure_batch_loss(state):
  return state['batch_train_loss']

def measure_test_acc(state):
  import tensorflow_datasets as tfds

  test_set = state['test_set']
  test_acc = AverageMeter()

  params = state['param_extractor'](state['opt_state'])
  acc_fun = state['acc_fun']

  for batch in tfds.as_numpy(test_set):
    batch_avg = acc_fun(params, batch).item()
    test_acc.update(batch_avg, len(batch['inputs']))

  return test_acc.val

def measure_l2_norm(state):
  from jax.experimental import optimizers

  params = state['param_extractor'](state['opt_state'])
  embed_params, rnn_params, output_params = params
  return optimizers.l2_norm(rnn_params)

