# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import yaml
import struct
import numpy as np

import tensorflow as tf

# from tensorflow.contrib.learn.python.learn.datasets import mnist

from chunkparser import ChunkParser
from train import FileDataSrc, get_latest_chunks

SKIP = 32
FLAGS = None
_NUMBER_OF_RECORDS=50000 # 197331


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  data = data_set.board
  # labels = data_set.labels
  # num_examples = data_set.num_examples
  num_examples = len(data)
  print('num_examples', num_examples)

  # if board.shape[0] != num_examples:
  #   raise ValueError('Images size %d does not match label size %d.' %
  #                    (board.shape[0], num_examples))
  # rows = images.shape[1]
  # cols = images.shape[2]
  # depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      
      print('{}: {:4} of {:4}'.format(name, index, num_examples))
      
      board = data[index]

      """
        int32 version (4 bytes)
        1858 float32 probabilities (7432 bytes)
        104 (13*8) packed bit planes of 8 bytes each (832 bytes)
        uint8 castling us_ooo (1 byte)
        uint8 castling us_oo (1 byte)
        uint8 castling them_ooo (1 byte)
        uint8 castling them_oo (1 byte)
        uint8 side_to_move (1 byte)
        uint8 rule50_count (1 byte)
        uint8 move_count (1 byte)
        int8 result (1 byte)
        planes <tf.Tensor 'Reshape:0' shape=(2048, 112, 64) dtype=float32>
        probs <tf.Tensor 'Reshape_1:0' shape=(2048, 1858) dtype=float32>
        winner <tf.Tensor 'Reshape_2:0' shape=(2048, 1) dtype=float32>
      """
      
      planes, probs, winner = board['planes'], board['probs'], board['winner']

      # planes = np.unpackbits(np.frombuffer(board['planes'], dtype=np.uint8)).astype(np.float32)
      # probs = np.frombuffer(board['probs'], dtype=np.uint8).astype(np.float32)
      # # probs = np.reshape(probs, (2048, 1858))

      lst_probs = []
      for idx in range(0, len(probs), 4):
          lst_probs.append(struct.unpack("f", probs[idx:idx+4])[0])
      probs = np.array(lst_probs)

      # if name == 'test':
      #   print('\n\nplanes:\n{}\n\n'.format(planes.shape))
      #   print('\n\nprobs:\n{}\n\n'.format(probs.shape))
      #   # print('\n\nwinner:\n{}\n\n'.format(winner))

      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'planes': _float_feature(planes),
                  'probs': _float_feature(probs),
                  'winner': _float_feature([winner])
                  
                  # 'planes': _bytes_feature(board.shape[2]),
                  # 'probabilities': _float_feature(board.shape[1]),
                  # 'us_ooo': _int64_feature(board.shape[3]),
                  # 'us_oo': _int64_feature(board.shape[4]),
                  # 'them_ooo': _int64_feature(board.shape[5]),
                  # 'them_oo': _int64_feature(board.shape[6]),
                  # 'side_to_move': _int64_feature(board.shape[7]),
                  # 'rule50_count': _int64_feature(board.shape[1]),
                  # 'move_count': _int64_feature(board.shape[1]),
                  # 'result': _int64_feature(board.shape[1])
              }))
      writer.write(example.SerializeToString())

def extract_data(parser: ChunkParser, chunkdata):
  lst = []
  gen = parser.sample_record(chunkdata)
  for s in gen:
    # v3_tuple = 
    (planes, probs, winner), (ver, probs2, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_plane, move_count, winner, planes1) = parser.convert_v3_to_tuple(s, return_planes=True)
    
    # planes, probs, winner = parser.parse_function(v3_tuple[0], v3_tuple[1], v3_tuple[2])
    shape = {'planes': planes1, 'probs': probs, 'winner': winner}
    lst.append(shape)
    # v3_tuple = parser.convert_v3_to_tuple(s, return_planes=True)
    # lst.append(v3_tuple)
  
  return lst

def read_data_sets(filenames,
                   fake_data=False,
                   one_hot=False,
                   dtype=tf.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
  
  cfg = yaml.safe_load(FLAGS.cfg.read())
  print(yaml.dump(cfg, default_flow_style=False))

  num_chunks = cfg['dataset']['num_chunks']
  train_ratio = cfg['dataset']['train_ratio']
  num_train = int(num_chunks*train_ratio)
  num_test = num_chunks - num_train
  if 'input_test' in cfg['dataset']:
    train_chunks = get_latest_chunks(cfg['dataset']['input_train'], num_train)
    test_chunks = get_latest_chunks(cfg['dataset']['input_test'], num_test)
  else:
    chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks)
    train_chunks = chunks[:num_train]
    test_chunks = chunks[num_train:]

  shuffle_size = cfg['training']['shuffle_size']
  total_batch_size = cfg['training']['batch_size']
  batch_splits = cfg['training'].get('num_batch_splits', 1)
  if total_batch_size % batch_splits != 0:
    raise ValueError('num_batch_splits must divide batch_size evenly')
  split_batch_size = total_batch_size // batch_splits
  # Load data with split batch size, which will be combined to the total batch size in tfprocess.
  ChunkParser.BATCH_SIZE = split_batch_size

  root_dir = os.path.join(cfg['training']['path'], cfg['name'])
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)

  t_chunks = FileDataSrc(train_chunks)
  train_parser = ChunkParser(t_chunks, shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
  
  final_train_data = []
  for chunkdata in t_chunks:
    if len(final_train_data) > _NUMBER_OF_RECORDS:
      break
    lst = extract_data(train_parser, chunkdata)
    for i in lst:
      print('{}: {:4}'.format('train', len(final_train_data)))
      final_train_data.append(i)


  shuffle_size = int(shuffle_size*(1.0-train_ratio))
  tt_chunks = FileDataSrc(test_chunks)
  test_parser = ChunkParser(tt_chunks, shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE)
  final_test_data = []
  
  for chunkdata in tt_chunks:
    if len(final_test_data) > _NUMBER_OF_RECORDS:
      break
    # print(filename)
    lst = extract_data(test_parser, chunkdata)
    for i in lst:
      print('{}: {:4}'.format('test', len(final_test_data)))
      final_test_data.append(i)
  
  train_parser.shutdown()
  test_parser.shutdown()
  
  op = type("Dataset", (object,), {
      'train': type('', (list, object), {'board': final_train_data})(),
      'test': type('', (list, object), {'board': final_test_data})()
    })()
  return op


def main(unused_argv):
  
  # Get the data.
  data_sets = read_data_sets(FLAGS.directory,
                                   dtype=tf.float32,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.test, 'test')
  # convert_to(data_sets.validation, 'validation')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/tmp/data',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  parser.add_argument('--cfg', type=argparse.FileType('r'), 
      help='yaml configuration with training parameters')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)