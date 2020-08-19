import numpy as np
from itertools import combinations
import functools
from sklearn.decomposition import PCA
import renn
from renn.rnn import cells
import jax.numpy as jnp
from jax.experimental import optimizers

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

def top_evecs(matrix_valued_fn, evaluation_pts, top_k = 2):
  """
  Returns the eigenvalues with the highest magnitude of eigenvectors
  for the given list of matrices
  """

  def top(m):
    R, E, L = renn.eigsorted(m)
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

# TODO there was a jax.jit here, should we do that?
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

  COLORS = ['red', 'blue', 'green', 'orange']

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

  COLORS = ['red', 'blue', 'green', 'orange']

  d0, d1 = pc_dimensions
  x_ro = readouts[:, d0]
  y_ro = readouts[:, d1]
  for k, states in states_by_value.items():
    # plot states
    for ind, trajectory in enumerate(states):
      if ind == 0:
        plt.plot(trajectory[:,d0], trajectory[:,d1], label=k, color=COLORS[k], alpha=0.2)
      else:
        plt.plot(trajectory[:,d0], trajectory[:,d1], color=COLORS[k], alpha=0.2)
      plt.scatter(trajectory[-1,d0], trajectory[-1,d1], marker='.', color=COLORS[k])

    # plot readouts
    plt.plot(np.array([0, x_ro[k]]), np.array([0,y_ro[k]]), color=COLORS[k], linestyle='dashed')

  # plot initial state
  plt.scatter(initial_state[0][d0], initial_state[0][d1],
              marker=(5,1), color='orange', s=400)

  plt.xlabel(f'PC Component {d0}')
  plt.ylabel(f'PC Component {d1}')
  plt.axis('equal')
  plt.legend(loc='upper right')

def plot_fp_2d(fixed_points, losses, readouts, initial_state, pc_dimensions,
               point_to_highlight=None):
  COLORS = ['red', 'blue', 'green', 'orange']

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
  LOGIT_COLORS = ['Reds', 'Blues', 'Greens', 'Oranges']
  prediction_color = ['red', 'blue', 'green', 'orange']

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

def pseudogrid(coordinates, dimension):
  all_coordinates = defaultdict(lambda: np.array(0.0))
  all_coordinates.update(coordinates)

  max_specified_dim = max(coordinates.keys())

  if max_specified_dim > 32:
    raise NotImplementedError('Maximum specified dimension cannot exceed 32')

  extra_dimensions = dimension - max_specified_dim - 1
  points = np.meshgrid(*[all_coordinates[i] for i in range(max_specified_dim+1)])
  points = np.stack(points).reshape(max_specified_dim+1, -1).T

  extra_coordinates = np.zeros((points.shape[0], extra_dimensions))

  return np.concatenate((points, extra_coordinates), axis=1)
