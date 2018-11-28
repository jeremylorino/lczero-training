#!/usr/bin/env python3
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

"""Converts data to TFRecords file format with Example protos."""
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

from chunkparser import ChunkParser
from train import FileDataSrc, get_latest_chunks

SKIP = 32
FLAGS = None

class Dataset(object):
  """
  Attributes:
    :ivar Dataset.Datalist train: training data
    :ivar Dataset.Datalist test: testing data
  """

  class Datalist(object):
    """
    Attributes:
      :ivar list board: data
    """
    def __init__(self, data=[]):
      self.board = data

  def __init__(self, train_data=[], test_data=[]):
    self.train = Dataset.Datalist(train_data)
    self.test = Dataset.Datalist(test_data)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_set: Dataset.Datalist, name):
  """
  Converts a dataset to tfrecords.
    :param Dataset.Datalist data_set: dataset to convert
    :param str name: data_set name
  """
  data = data_set.board
  num_examples = len(data)

  tf.logging.info('Number of {} examples: {}'.format(name, num_examples))

  filename = os.path.join(FLAGS.gcs_bucket, name + '.tfrecords')
  
  tf.logging.info('Writing {}'.format(filename))

  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      
      tf.logging.debug('{}: {:4} of {:4}'.format(name, index, num_examples))
      
      board = data[index]
      planes, probs, winner = board['planes'], board['probs'], board['winner']
      lst_probs = []
      for idx in range(0, len(probs), 4):
          lst_probs.append(struct.unpack("f", probs[idx:idx+4])[0])
      probs = np.array(lst_probs)

      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'planes': _float_feature(planes),
                  'probs': _float_feature(probs),
                  'winner': _float_feature([winner])
              }))
      writer.write(example.SerializeToString())

def extract_data(parser: ChunkParser, chunkdata):
  lst = []
  gen = parser.sample_record(chunkdata)
  for s in gen:
    (planes, probs, winner), (ver, probs2, planes, us_ooo, us_oo, them_ooo, them_oo, stm, rule50_plane, move_count, winner, planes1) = parser.convert_v3_to_tuple(s, return_planes=True)
    
    shape = {'planes': planes1, 'probs': probs, 'winner': winner}
    lst.append(shape)
  
  return lst

def read_data_sets(filenames,
                   cfg=None,
                   fake_data=False,
                   one_hot=False,
                   dtype=tf.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):
  
  if cfg is None:
    cfg = yaml.safe_load(FLAGS.cfg.read())
  tf.logging.info(yaml.dump(cfg, default_flow_style=False))

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
  train_parser = ChunkParser(t_chunks, shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE, auto_start_workers=False)
  final_train_data = []
  
  tf.logging.info('Loading training dataset')

  for chunkdata in t_chunks:
    if len(final_train_data) > FLAGS.record_count:
      break
    lst = extract_data(train_parser, chunkdata)
    for i in lst:
      tf.logging.debug('{}: {:4}'.format('train', len(final_train_data)))
      final_train_data.append(i)

  shuffle_size = int(shuffle_size*(1.0-train_ratio))
  tt_chunks = FileDataSrc(test_chunks)
  test_parser = ChunkParser(tt_chunks, shuffle_size=shuffle_size, sample=SKIP, batch_size=ChunkParser.BATCH_SIZE, auto_start_workers=False)
  final_test_data = []
  
  tf.logging.info('Loading testing dataset')
  
  for chunkdata in tt_chunks:
    if len(final_test_data) > FLAGS.record_count:
      break
    lst = extract_data(test_parser, chunkdata)
    for i in lst:
      tf.logging.debug('{}: {:4}'.format('test', len(final_test_data)))
      final_test_data.append(i)
  
  train_parser.shutdown()
  test_parser.shutdown()
  
  datasets = Dataset(train_data=final_train_data, test_data=final_test_data)
  return datasets

def main(unused_argv):
  
  cfg = yaml.safe_load(FLAGS.cfg.read())

  # Get the data.
  data_sets = read_data_sets(FLAGS.gcs_bucket,
                                   cfg = cfg,
                                   dtype=tf.float32,
                                   reshape=False)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.test, 'test')
  ## currently not supporting a validation dataset
  # convert_to(data_sets.validation, 'validation')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--gcs_bucket',
      default=os.environ.get('GCS_BUCKET'),
      type=str,
      help='Google Cloud Storage bucket path to write converted results. (defaults to environment variable \'GCS_BUCKET\')'
  )
  parser.add_argument(
      '--record_count',
      type=int,
      default=50000,
      help='Number of examples to add to each dataset'
  )
  parser.add_argument('--cfg', type=argparse.FileType('r'), 
      help='yaml configuration with training parameters')
  
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      '--log_level',
      type=str,
      choices=['debug', 'error', 'fatal', 'info', 'warn'],
      default='info',
      help='Logging verbosity'
  )
  group.add_argument(
      '--verbose',
      action='store_true',
      help='Verbose output'
  )

  FLAGS, unparsed = parser.parse_known_args()

  if FLAGS.verbose:
    tf.logging.set_verbosity(tf.logging.DEBUG)
  tf.logging.set_verbosity(getattr(tf.logging, FLAGS.log_level.upper(), tf.logging.INFO))

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)