import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def cumulative_var_explained(pca_object):
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
