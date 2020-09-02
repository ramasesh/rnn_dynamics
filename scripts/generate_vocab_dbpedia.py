import csv
import renn
from google.cloud import storage

vocab_size=50000
vocab_filename = 'dbpedia_vocab.csv'

with open('data/dbpedia/train.csv', 'r') as csvfile:
  print('Successfully opened file')

client = storage.Client()
bucket = client.get_bucket('ramasesh-bucket-1')
blob = bucket.blob('test_file_dbpedia')
blob.upload_from_filename('requirements.txt')

print('Successfully sent to bucket')


with open('data/dbpedia/train.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  two_reader = map(lambda x: x[2], reader)
  vocab = renn.data.build_vocab(two_reader, vocab_size,
                                split_fun = lambda d: d.lower().split())

with open(vocab_filename, 'w') as f:
  for v in vocab:
    f.write(v + '\n')

client = storage.Client()
bucket = client.get_bucket('ramasesh-bucket-1')
blob = bucket.blob('dbpedia_vocab')
blob.upload_from_filename(vocab_filename)
