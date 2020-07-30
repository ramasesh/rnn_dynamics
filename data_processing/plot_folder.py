import matplotlib.pyplot as plt

from absl import flags, app
import json_lines
import os

FLAGS = flags.FLAGS

BASEDIR='results'

flags.DEFINE_string('directory', None, 'Directory with the data')

def main(_):
  full_dir = os.path.join(BASEDIR, FLAGS.directory)
  all_files = [f for f in os.listdir(full_dir) if f[-5:] == 'jsonl']
  print(all_files)

  data_to_plot = {}
  for filename in all_files:
    data_to_plot[filename] = {'step': [], 'value': []}
    with open(os.path.join(full_dir, filename), 'rb') as f:
      for item in json_lines.reader(f):
        data_to_plot[filename]['step'].append(int(item['step']))
        data_to_plot[filename]['value'].append(float(item['value']))
  
  f = plt.figure(figsize=(10,10))
  for k in data_to_plot.keys():
    if 'test' in k:
      plt.plot(data_to_plot[k]['step'], data_to_plot[k]['value'], label=k[:-6])
    else:
      plt.scatter(data_to_plot[k]['step'], data_to_plot[k]['value'], label=k[:-6])
  plt.legend(loc=3)
  plt.xlabel("Step")
  plt.title(FLAGS.directory)
  plt.ylabel("Loss/Accuracy")
  plt.savefig(os.path.join(full_dir, 'training_plot.png'))


if __name__ == '__main__':
  app.run(main)
  
