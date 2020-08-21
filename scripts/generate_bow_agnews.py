import pickle
import renn

import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
from jax.experimental import stax, optimizers

import numpy as np

from renn.losses import multiclass_xent
from renn import serialize
from functools import partial

vocab_file='./data/vocab/ag_news.vocab'

seq_length = 160
num_classes = 3

def filter_fn(item):
  return item['labels'] < num_classes

train_dset = renn.data.ag_news('train', vocab_file, sequence_length=seq_length,
                               data_dir='./data', filter_fn=filter_fn)
test_dset = renn.data.ag_news('test', vocab_file, sequence_length=seq_length,
                              data_dir='./data', filter_fn=filter_fn)

with open(vocab_file, 'r') as f:
  vocab = f.readlines()
vocab_size=len(vocab)

example = next(iter(train_dset))

def SequenceSum():
  def init_fun(_, input_shape):
    return (input_shape[0], input_shape[2]), ()
  def apply_fun(_, inputs, **kwargs):
    return jnp.sum(inputs, axis=1)
  return init_fun, apply_fun

emb_size = 32

input_shape = (-1, seq_length)
l2_pen = 0

init_fun, apply_fun = stax.serial(
    renn.embedding(vocab_size, emb_size),
    SequenceSum(),
    stax.Dense(num_classes),
)

key = jax.random.PRNGKey(0)
output_shape, initial_params = init_fun(key, input_shape)

emb = initial_params[0]
new_emb = np.array(emb)
new_emb[0] = np.zeros(emb_size)
initial_params = [jnp.array(new_emb), *initial_params[1:]]

def xent(params, batch):
  logits = apply_fun(params, batch['inputs'])
  data_loss = multiclass_xent(logits, batch['labels'])
  reg_loss = l2_pen * renn.norm(params)
  return data_loss + reg_loss

f_df = jax.value_and_grad(xent)

@jax.jit
def accuracy(params, batch):
  logits = apply_fun(params, batch['inputs'])
  predictions = jnp.argmax(logits, axis=1)
  return jnp.mean(predictions == batch['labels'])

learning_rate = optimizers.exponential_decay(2e-3, 1000, 0.8)
init_opt, update_opt, get_params = optimizers.adam(learning_rate)

state = init_opt(initial_params)
losses = []

@jax.jit
def step(k, opt_state, batch):
  params = get_params(opt_state)
  loss, gradients = f_df(params, batch)
  new_state = update_opt(k, gradients, opt_state)
  return new_state, loss

def test_acc(params):
  return jnp.array([accuracy(params, batch) for batch in tfds.as_numpy(test_dset)])

for epoch in range(1):
  print('======')
  print(f'== Epoch {epoch}')
  p = get_params(state)
  acc = np.mean(test_acc(p))
  print(f'== Test accuracy: {100. * acc:0.2f}')
  print('======')

  for batch in tfds.as_numpy(train_dset):
    k = len(losses)
    state, loss = step(k, state, batch)
    losses.append(loss)

    if k % 100 == 0:
      p = get_params(state)
      print(f'[step {k}]\tLoss: {np.mean(losses[k-100:k]):0.4f}', flush=True)

print('======')
print(f'== Epoch {epoch}')
p = get_params(state)
acc = np.mean(test_acc(p))
print(f'== Test accuracy: {100. * acc:0.2f}')
print('======')

params = get_params(state)

with open('bow_agnews_3class', 'wb') as f:
  pickle.dump(params, f)
