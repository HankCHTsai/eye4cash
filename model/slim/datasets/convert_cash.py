# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Converts Cash data to TFRecords of TF-Example protos.

This module reads the Cash dataset files and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf
import numpy as np

from datasets import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 257

# Seed for repeatability.
_RANDOM_SEED = 0

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_filenames_and_labels(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  cash_label = os.path.join(dataset_dir, 'raw/label.txt')
  filenames = []
  labels = []
 
  if not tf.gfile.Exists(cash_label):
    return filenames, labels

  with open(cash_label, "r") as labelfile:
    #line_indx = 1
    for columns in [line.split() for line in labelfile]:
      #line_indx += 1
      #if len(columns) < 14:
      #  sys.stdout.write('\r>> Label error in line %d' % (line_indx))
      #    sys.stdout.flush()
      #  continue
      filenames.append('%s/raw/%s' % (dataset_dir, columns[0]))
      labels.append(int(columns[1]))
      #labels.append('%s %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % (columns[1], columns[2], columns[3], columns[4],
      #  columns[5], columns[6], columns[7], columns[8], columns[9], columns[10], columns[11], columns[12], columns[13]))

  return filenames, labels


def _get_dataset_filename(dataset_dir, orig_filepath, split_name):
  output_filename = 'cash_%s_%s.tfrecord' % (
      os.path.basename(orig_filepath), split_name)
  return dataset_dir + 'raw/' + output_filename
  #return os.path.join(dataset_dir, 'raw/' + output_filename)


def _convert_dataset(split_name, files, labels, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for file_indx in range(len(files)):
        output_filename = _get_dataset_filename(
            dataset_dir, files[file_indx], split_name)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          sys.stdout.write('\r>> Converting image %d/%d' % (
              file_indx, len(files)))
          sys.stdout.flush()

          # Read the filename:
          image_data = tf.gfile.FastGFile(files[file_indx], 'rb').read()
          #image_raw = image_reader.decode_jpeg(sess, image_data)
          #height, width = image_raw.shape[0], image_raw.shape[1]
          height, width = image_reader.read_image_dims(sess, image_data)

          example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, labels[file_indx])
          tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  #if not tf.gfile.Exists(dataset_dir):
  #  return False
  #return True
  return False


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  photo_filenames, labels = _get_filenames_and_labels(dataset_dir)

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]
  training_labels = labels[_NUM_VALIDATION:]
  validation_labels = labels[:_NUM_VALIDATION]
  print(len(training_filenames))
  print(len(validation_filenames))

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, training_labels,
                   dataset_dir)
  _convert_dataset('validation', validation_filenames, validation_labels,
                   dataset_dir)

  print('\nFinished converting the Cash dataset!')

