import jax.numpy as jnp
import numpy as np

num_classes = 3

vocab = {'very0': jnp.array([2,0,0]),
         'very1': jnp.array([0,2,0]),
         'very2': jnp.array([0,0,2]),
         'some0': jnp.array([1,0,0]),
         'some1': jnp.array([0,1,0]),
         'some2': jnp.array([0,0,1]),
         'neutral0': jnp.array([0,0,0]),
         'neutral1': jnp.array([1,1,1]),
         'not0': jnp.array([-1,0,0]),
         'not1': jnp.array([0,-1,0]),
         'not2': jnp.array([0,0,-1])}

numerical_vocab = {0: jnp.array([2,0,0]),
                   1: jnp.array([0,2,0]),
                   2: jnp.array([0,0,2]),
                   3: jnp.array([1,0,0]),
                   4: jnp.array([0,1,0]),
                   5: jnp.array([0,0,1]),
                   6: jnp.array([0,0,0]),
                   7: jnp.array([1,1,1]),
                   8: jnp.array([-1,0,0]),
                   9: jnp.array([0,-1,0]),
                   10: jnp.array([0,0,-1])}

def uniform_length_sampler(length):

  def sample(num):
    return jnp.array([length]*num)

  return sample


def variable_length_sampler(mean, std):

  MIN_LENGTH = 20
  MAX_LENGTH = 100
  def sample(num):
    return np.round(np.clip(np.random.randn(num)*std + mean, MIN_LENGTH, MAX_LENGTH)).astype(int)

  return sample

def ground_truth_label(sentences, lengths):

  scores = batch_score(sentences, lengths)
  return jnp.argmax(scores, axis=1)

def score(sentence, length):

  return sum([numerical_vocab[si] for si in sentence[:length]])

def batch_score(sentences, lengths):

  return jnp.array([score(s, l) for s,l in zip(sentences, lengths)])

def sample_batch(batch_size=64, length_sampler = None):

  if length_sampler == None:
    DEFAULT_LENGTH = 20
    length_sampler = uniform_length_sampler(DEFAULT_LENGTH)

  lengths = length_sampler(num=batch_size)
  MAX_LENGTH = 100

  num_tokens = len(numerical_vocab)
  batch = {'inputs': np.random.randint(num_tokens, size=(batch_size, MAX_LENGTH))}

  batch['labels'] = ground_truth_label(batch['inputs'], lengths)
  batch['index'] = lengths

  return batch

class ToyDataset:

  def __init__(self, mean_length, std_length, batch_size=64):
    self.mean_length = mean_length
    self.std_length = std_length
    self.batch_size = batch_size

    self.sampler = variable_length_sampler(mean_length, std_length)

  def __iter__(self):
    return self

  def __next__(self):
    return sample_batch(self.batch_size, self.sampler)
