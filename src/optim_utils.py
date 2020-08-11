import jax
import jax.numpy as jnp
from jax.experimental import optimizers

from renn import utils, losses

def build_optimizer_step(optimizer, initial_params, loss_fun, gradient_clip=None):
  """Builds training step function."""

  #Destructure the optimizer triple.
  init_opt, update_opt, get_params = optimizer
  opt_state = init_opt(initial_params)

  if gradient_clip is None:
    gradient_clip = jnp.inf

  @jax.jit
  def optimizer_step(current_step, state, batch):
    """Takes a single optimization step."""
    p = get_params(state)
    loss, gradients = jax.value_and_grad(loss_fun)(p, batch)

    gradients = optimizers.clip_grads(gradients, gradient_clip)

    new_state = update_opt(current_step, gradients, state)
    return current_step + 1, new_state, loss

  return opt_state, optimizer_step

def get_optimizer(optim_config):
  """ returns an ADAM optimizer with exponential learning-rate
  decay schedule specified in the config """

  learning_rate = optimizers.exponential_decay(optim_config['base_lr'],
                                               optim_config['lr_decay_steps'],
                                               optim_config['lr_decay_rate'])
  opt = optimizers.adam(learning_rate)

  return opt

def optimization_suite(initial_params, loss_function, optim_config):

  optimizer = get_optimizer(optim_config)
  get_params = optimizer[2]

  optimizer_state, step_function = build_optimizer_step(optimizer,
                                                        initial_params,
                                                        loss_function,
                                                        optim_config['gradient_clip'])

  return optimizer, get_params, optimizer_state, step_function

def loss_and_accuracy(network_fun, model_config, optim_config):

  n_out = model_config['num_outputs']
  bare_loss = losses.binary_xent if n_out == 1 else losses.multiclass_xent

  l2_reg = l2_loss(optim_config['L2'])

  def loss_fun(params, batch):
    all_time_logits = network_fun(params, batch['inputs'])
    end_logits = utils.select(all_time_logits, batch['index'])

    return bare_loss(end_logits, batch['labels']) + l2_reg(params)

  if n_out == 1:
    prediction_function = lambda x: (x >= 0.).astype(jnp.int32)
  else:
    prediction_function = lambda x: x.argmax(axis=-1).astype(jnp.int32)

  def acc_fun(params, batch):
    all_time_logits = network_fun(params, batch['inputs'])
    end_logits = utils.select(all_time_logits, batch['index'])
    predictions = jnp.squeeze(prediction_function(end_logits))
    accuracies = (batch['labels'] == predictions).astype(jnp.int32)
    return jnp.mean(accuracies)

  return loss_fun, acc_fun

def l2_loss(l2_penalty):
  """ Returns a loss function which maps parameters to
  the l2_penalty times the l2_norm of the RNN parameters only"""

  if l2_penalty == 0.0:
    return lambda x: 0.0
  else:
    def l2(params):
      emb_params, rnn_params, readout_params = params
      return l2_penalty * optimizers.l2_norm(rnn_params)

    return l2
