import matplotlib.pyplot as plt

from absl import flags, app
import json_lines
import os

FLAGS = flags.FLAGS

BASEDIR='results'

flags.DEFINE_string('directory', None, 'Directory with the data')
flags.DEFINE_bool('datadir', False, 'If True, then will assume this data has the jsonl files') 

def process_data_dir(folder):
  all_files = [f for f in os.listdir(folder) if f[-5:] == 'jsonl']
  print(all_files)

  data_to_plot = {}
  for filename in all_files:
    data_to_plot[filename] = {'step': [], 'value': []}
    with open(os.path.join(folder, filename), 'rb') as f:
      for item in json_lines.reader(f):
        data_to_plot[filename]['step'].append(int(item['step']))
        data_to_plot[filename]['value'].append(float(item['value']))

  f, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
  for k in data_to_plot.keys():
    if 'acc' in k:
      ax[0].plot(data_to_plot[k]['step'], data_to_plot[k]['value'], label=k[:-6])
    elif 'loss' in k:
      ax[1].scatter(data_to_plot[k]['step'], data_to_plot[k]['value'], label=k[:-6])
    else:
      ax[2].scatter(data_to_plot[k]['step'], data_to_plot[k]['value'], label=k[:-6])
  ax[0].legend()
  ax[1].legend()
  ax[2].legend()

  plt.xlabel("Step")
  f.suptitle(folder)
  plt.ylabel("Loss/Accuracy")
  plt.savefig(os.path.join(folder, 'training_plot.png'))

def main(_):
  full_dir = os.path.join(BASEDIR, FLAGS.directory)

  if FLAGS.datadir:
    process_data_dir(full_dir)
  else:
    print(os.listdir(full_dir))
    folders = [f for f in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir,f))]
    print(folders)
    for folder in folders:
      process_data_dir(os.path.join(full_dir, folder))

if __name__ == '__main__':
  app.run(main)
  
