import numpy as np
from itertools import combinations
import functools
from sklearn.decomposition import PCA
import renn
from renn.rnn import cells
import jax.numpy as jnp
from jax.experimental import optimizers
import jax
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cumulative_var_explained(pca_object):
  """ For a given PCA object, returns how much variance
  is explained by each dimension """
  return np.cumsum(pca_object.explained_variance_ratio_)

def plot_varexp(var_exp_dictionary):

  fig = plt.figure(figsize=(8,4))
  for k, v in var_exp_dictionary.items():
    plt.scatter(np.arange(len(v)), v, label=k)

  plt.xlabel('Dimension')
  plt.ylabel('Variance explained')
  plt.ylim(0, 1.1)
  plt.xlim(-1, 20)
  plt.legend(loc='lower right')

  plt.grid()

  return fig

def alignment(vecs_1, vecs_2):
  """ returns dot-product overlap between vecs_1 and vecs_2 """
  return np.diag(np.dot(vecs_1, vecs_2.T))

def get_alignment_dictionary(PCA_dictionary):
  """ returns the alignments between all of the pairs of PCA axes in
  PCA dictionary """

  component_vectors = {k: v.components_ for k,v in PCA_dictionary.items()}

  def alignment_by_name(k1, k2):
    return alignment(component_vectors[k1], component_vectors[k2])

  alignment_dictionary = {combined_k: alignment_by_name(*combined_k) for combined_k in combinations(component_vectors.keys(), 2)}

  return alignment_dictionary

def plot_alignment(PCA_dictionary):

  alignment_dictionary = get_alignment_dictionary(PCA_dictionary)
  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))

  for ind, k in enumerate(alignment_dictionary.keys()):
    alignments = alignment_dictionary[k]

    ax[ind].scatter(np.arange(len(alignments)), np.abs(alignments))
    ax[ind].set_title(k)
    ax[ind].set_xlim(0,20)
    ax[ind].set_ylim(-.1, 1.1)
    ax[ind].grid()
    ax[ind].set_xlabel('PCA_direction')

  fig.tight_layout()
  return fig

def eigsorted(jac):
  unsorted_eigvals, unsorted_rights = np.linalg.eig(jac)

  def sorting_function(evals):
    return np.abs(1. /np.log(np.abs(evals)))

  sorted_indices = np.flipud(np.argsort(sorting_function(unsorted_eigvals)))

  eigenvalues = unsorted_eigvals[sorted_indices]
  rights = unsorted_rights[:, sorted_indices]
  lefts = np.linalg.pinv(rights).T

  return rights, eigenvalues, lefts

def top_evecs(matrix_valued_fn, evaluation_pts, top_k = 2):
  """
  Returns the eigenvalues with the highest magnitude of eigenvectors
  for the given list of matrices
  """

  def top(m):
    R, E, L = eigsorted(m)
    top_eigs = R[:,:top_k]

    return top_eigs

  return np.array([top(matrix_valued_fn(p)) for p in evaluation_pts])

class PCA_rnn(PCA):
  def __init__(self):
    super().__init__()

  def transform_no_mean(self, X):
    if self.mean_ is not None:
      temp_mean = self.mean_
      self.mean_ = None
      X_transformed = np.dot(X, self.components_.T)
      self.mean_ = temp_mean

      return X_transformed
    else:
      return self.transform(X)

def _get_all_states(cell, inputs, rnn_params, emb_params, emb_apply_fn):
  rnn_inputs = emb_apply_fn(emb_params, inputs)
  initial_states = cell.get_initial_state(rnn_params,
                                           batch_size=rnn_inputs.shape[0])
  return renn.unroll_rnn(initial_states,
                         rnn_inputs,
                         functools.partial(cell.batch_apply, rnn_params))

def rnn_states(cell, batch, rnn_params, emb_params, emb_apply_fn):
  states = _get_all_states(cell, batch['inputs'], rnn_params, emb_params, emb_apply_fn)
  return [h[:idx] for h, idx in zip(states, batch['index'])]

def rnn_end_states(cell, batch, rnn_params, emb_params, emb_apply_fn):
  states = rnn_states(cell, batch, rnn_params, emb_params, emb_apply_fn)
  return [h[-1].reshape(1,-1) for h in states]

def states_by_value(states, labels, num_classes):
  return {k: [h for h, lbl in zip(states, labels) if lbl == k]
            for k in range(num_classes)}

def plot_states(states_by_value, readouts=None,
                initial_state=None, pc_dimensions=[0,1]):

  if len(pc_dimensions) == 2:
    return plot_states_2d(states_by_value, readouts, initial_state, pc_dimensions)
  elif len(pc_dimensions) == 3:
    return plot_states_3d(states_by_value, readouts, initial_state, pc_dimensions)
  else:
    raise ValueError('pc_dimensions must have length 2 or 3')

def plot_states_3d(states_by_value, readouts, initial_state, pc_dimensions):

  COLORS = ['red', 'blue', 'green', 'orange', 'purple']

  d0, d1, d2 = pc_dimensions
  x_ro = readouts[:, d0]
  y_ro = readouts[:, d1]
  z_ro = readouts[:, d2]

  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(111, projection='3d')
  for k, states in states_by_value.items():
    # plot states
    for ind, trajectory in enumerate(states):
      if ind == 0:
        ax.plot(trajectory[:,d0], trajectory[:,d1], trajectory[:,d2], label=k, color=COLORS[k], alpha=0.2)
      else:
        ax.plot(trajectory[:,d0], trajectory[:,d1], trajectory[:,d2], color=COLORS[k], alpha=0.2)
      ax.scatter(trajectory[-1,d0], trajectory[-1,d1], trajectory[-1,d2], marker='.', color=COLORS[k])

    # plot readouts
    ax.plot(np.array([0, x_ro[k]]), np.array([0,y_ro[k]]), np.array([0,z_ro[k]]),
            color=COLORS[k], linestyle='dashed')

  # plot initial state
  ax.scatter(initial_state[0][d0], initial_state[0][d1],
              initial_state[0][d2],
              marker=(5,1), color='orange', s=400)

  return fig, ax

  #plt.xlabel(f'PC Component {d0}')
  #plt.ylabel(f'PC Component {d1}')
  #plt.axis('equal')
  #plt.legend(loc='upper right')

def plot_states_2d(states_by_value, readouts, initial_state, pc_dimensions):

  COLORS = ['red', 'blue', 'green', 'orange', 'purple']

  d0, d1 = pc_dimensions
  x_ro = readouts[:, d0]
  y_ro = readouts[:, d1]
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(111)
  for k, states in states_by_value.items():
    # plot states
    for ind, trajectory in enumerate(states):
      if ind == 0:
        ax.plot(trajectory[:,d0], trajectory[:,d1], label=k, color=COLORS[k], alpha=0.2)
      else:
        ax.plot(trajectory[:,d0], trajectory[:,d1], color=COLORS[k], alpha=0.2)
      ax.scatter(trajectory[-1,d0], trajectory[-1,d1], marker='.', color=COLORS[k])

    # plot readouts
    ax.plot(np.array([0, x_ro[k]]), np.array([0,y_ro[k]]), color=COLORS[k], linestyle='dashed')

  # plot initial state
  ax.scatter(initial_state[0][d0], initial_state[0][d1],
              marker=(5,1), color='orange', s=400)

  ax.set_xlabel(f'PC Component {d0}')
  ax.set_ylabel(f'PC Component {d1}')
  ax.legend(loc='upper right')

  return fig, ax

def plot_traj_2d(trajectories, pc_dimensions, initial_state=None, labels=None,
                 xlim=4, ylim=4):

  COLORS = ['red', 'blue', 'green', 'orange', 'purple']

  d0, d1 = pc_dimensions
  for traj_ind, traj in enumerate(trajectories):
    # plot states
    if labels is not None:
      plt.plot(traj[:,d0], traj[:,d1], color=COLORS[labels[traj_ind]], alpha=0.2)
    else:
      plt.plot(traj[:,d0], traj[:,d1], color='k', alpha=0.2)
    plt.scatter(traj[-1,d0], traj[-1,d1], marker='.', color='k')

  if initial_state is not None:
    # plot initial state
    plt.scatter(initial_state[0][d0], initial_state[0][d1],
              marker=(5,1), color='orange', s=400)

  plt.xlim(-xlim,xlim)
  plt.ylim(-ylim,ylim)
  plt.xlabel(f'PC Component {d0}')
  plt.ylabel(f'PC Component {d1}')

def plot_fp_2d(fixed_points, losses, readouts, initial_state, pc_dimensions,
               point_to_highlight=None):
  COLORS = ['red', 'blue', 'green', 'orange', 'purple']

  d0, d1 = pc_dimensions
  x_ro = readouts[:, d0]
  y_ro = readouts[:, d1]

  x = fixed_points[:, d0]
  y = fixed_points[:, d1]

  plt.scatter(x, y, s=20, c = np.log10(losses))
  if point_to_highlight is not None:
    plt.scatter(x[point_to_highlight],
                y[point_to_highlight],
                s=100,
                c='red',
                marker='+')
  plt.grid()

  for k in range(len(readouts)):
    # plot readouts
    plt.plot(np.array([0, x_ro[k]]), np.array([0,y_ro[k]]), color=COLORS[k], linestyle='dashed',label=k)

  # plot initial state
  plt.scatter(initial_state[0][d0], initial_state[0][d1],
              marker=(5,1), color='orange', s=400)

  plt.xlabel(f'PC Component {d0}')
  plt.ylabel(f'PC Component {d1}')
  plt.axis('equal')
  plt.legend(loc='upper right')

def fixed_points(cell,
                 rnn_params,
                 initial_states,
                 tolerance,
                 embedding_size,
                 noise_scale=0.0,
                 learning_rate=0.01,
                 decimation_factor=1):

  if isinstance(initial_states, list):
    initial_states = np.vstack(initial_states)
  initial_states = initial_states[::decimation_factor]
  initial_states += noise_scale * np.random.randn(*initial_states.shape)

  input_lin = jnp.zeros((initial_states.shape[0], embedding_size))

  fp_loss_fn = renn.build_fixed_point_loss(cell, rnn_params)
  fixed_points, loss_hist, fp_losses = renn.find_fixed_points(fp_loss_fn,
                                                              initial_states,
                                                              input_lin,
                                                              optimizers.adam(0.01),
                                                              tolerance)

  return fixed_points, loss_hist, fp_losses

def plot_logits(readout_function, readout_parameters, point_set,
                PCA_object, pc_dimensions=[0,1]):
  LOGIT_COLORS = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']
  prediction_color = ['red', 'blue', 'green', 'orange', 'purple']

  logits = readout_function(readout_parameters, point_set)

  n_logits = logits.shape[1]
  predictions = np.argmax(logits, axis=1)
  biases = readout_parameters[1]

  transformed_pts = PCA_object.transform(point_set).T
  x = transformed_pts[pc_dimensions[0]]
  y = transformed_pts[pc_dimensions[1]]

  f, ax = plt.subplots(nrows=1, ncols=n_logits+1, figsize=(4*(n_logits+1), 4))

  if n_logits == 1:
    ax = [ax]

  for ind, a in enumerate(ax[:-1]):
    p = a.scatter(x, y, c=logits[:,ind], cmap=LOGIT_COLORS[ind])
    divider = make_axes_locatable(a)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(p, cax=cax)

    b = biases[ind]
    a.set_title(f'Logit {ind}, Bias {b:0.2f}')
  
  ax[-1].scatter(x, y, c=[prediction_color[pi] for pi in predictions])

  f.suptitle('Logit values among fixed points')
  f.tight_layout()
  return f

from collections import defaultdict

def squash(points, transform_object, n_dims_keep):
  """ Given a set of points and a transform object,
  transform the points using the transform, squash all
  dimensions after the first n_dims_keep dimensions to
  zero, and then transforms back to original space """

  transformed = transform_object.transform(points)
  transformed[:, n_dims_keep:] = 0.0
  squashed = transform_object.inverse_transform(transformed)
  return squashed

def shuffle_words(batch):
  """ Returns a batch in which each example is a shuffled
  version of the sentences in the argument batch. """

  batch.update({'inputs': shuffle(batch['inputs'], batch['index'])})
  return batch

def shuffle(sentences, lengths):
  """ Shuffles sentences, respecting the length of the
  sentences """
  n_sentences, sentence_length = sentences.shape
  permuted_indices = np.full(sentences.shape,
                             np.arange(sentence_length))

  for i in range(n_sentences):
    permutation = np.random.permutation(lengths[i])
    permuted_indices[i, :len(permutation)] = permutation

  return reorder(sentences, permuted_indices)

def reorder(input_array, indices):
  """ Reorders row of input_array according to the specification
  in indices

  Arguments:
    input_array: a 2D np array
    indices: a np array with the same shape as input_array

  Returns:
    reordered_array: a 2D np array whose rows have been
                     reordered from input_array

  Example:
    input_array: np.array([['A', 'B', 'C'],
                           ['D', 'E', 'F']])
    indices: np.array([[0,1,2],
                       [2,1,0]])
    reordered_array: np.array([['A', 'B', 'C'],
                               ['F', 'E', 'D']])
  """
  reordered_array = np.array([row[order] for row, order in zip(input_array, indices)])
  return reordered_array
