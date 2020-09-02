from absl import flags, app
import glob
import json
from src import data, model_utils, measurements
import numpy as np
from data_processing import analysis_utils as au
import renn
from renn.rnn import network
import json_lines
import os
import tensorflow_datasets as tfds

from jax.config import config as c
c.update('jax_disable_jit', True)

FLAGS = flags.FLAGS

flags.DEFINE_enum('epochs', '2', ['2','5'], 'n epochs')
flags.DEFINE_enum('arch','LSTM', ['LSTM', 'GRU', 'UGRNN'], 'architecture')
flags.DEFINE_enum('l2', '1e-06', ['1e-06', '1e-05', '0.0001', '0.001', '0.01', '0.1'], 'l2 penalty')
flags.DEFINE_enum('eta', '0.001', ['0.001', '0.003', '0.01', '0.03', '0.1'], 'learning rate')
flags.DEFINE_boolean('shuffle', False, 'shufles dataset')

def main(_):
  BASE_FOLDER = f'results/yelp/jointsweep/{FLAGS.epochs}Epochs/{FLAGS.arch}_eta_{FLAGS.eta}_L2_{FLAGS.l2}_*'
  data_folder = glob.glob(BASE_FOLDER)

  assert len(data_folder) == 1
  data_folder = data_folder[0]

  with open(os.path.join(data_folder, 'config.json')) as f:
    config = json.load(f)
  
  with open(os.path.join(data_folder, 'test_acc.jsonl')) as f:
    x = json_lines.reader(f)
    print("Non shuffled acc (recorded):")
    print(list(x)[-1]['value'])

  vocab_size, train_dset, test_dset = data.get_dataset(config['data'])

  cell = model_utils.get_cell(config['model']['cell_type'],
                              num_units=config['model']['num_units'])
  init_fun, apply_fun, emb_apply, readout_apply = network.build_rnn(vocab_size,
                                                                    config['model']['emb_size'],
                                                                    cell,
                                                                    num_outputs=config['model']['num_outputs'])
  emb_init, emb_apply = renn.embedding(vocab_size,
                                       config['model']['emb_size'])
  network_params = model_utils.load_params(os.path.join(data_folder, 'final_params'))
  emb_params, rnn_params, readout_params = network_params

  print("Loaded model and dataset")

  test_acc = measurements.AverageMeter()
  for i, batch in enumerate(tfds.as_numpy(test_dset)):
    if FLAGS.shuffle:
      batch = au.shuffle_words(batch)
    batch_final_states = au.rnn_end_states(cell,
                                           batch,
                                           rnn_params,
                                           emb_params,
                                           emb_apply)
    print(i)
    """
    logits = readout_apply(readout_params, np.vstack(batch_final_states))
    predictions = np.argmax(logits, axis=1)

    curr_acc = np.mean(predictions == batch['labels'])
    test_acc.update(curr_acc, len(batch['index']))

    print(i, len(batch['index']))

    del batch_final_states
    del logits
    del predictions
    del batch
    """

    #if i > 85:
    #  break

  if FLAGS.shuffle:
    print("Shuffled accuracy")
  else:
    print("Non-shuffled accuracy")
  print(test_acc.avg)

if __name__ == '__main__':
  app.run(main)
