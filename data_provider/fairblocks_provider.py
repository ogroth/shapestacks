"""
Provides an input_fn for tf.estimator.Estimator to load the images of the real
world FAIR block tower test set presented in https://arxiv.org/abs/1603.01312:
  Learning Physical Intuition of Block Towers by Example
  Adam Lerer, Sam Gross, Rob Fergus
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np


# dataset constants
_CHANNELS = 3 # RGB images
_HEIGHT = 224
_WIDTH = 224
_NUM_CLASSES = 2 # stable | unstable
# label semantics: 0 = stable | 1 = unstable


# internal dataset creation, file parsing and pre-processing

def _get_filenames_with_labels(mode, data_dir, split_dir):
  """
  Returns all training or test files in the data directory with their
  respective labels.
  """
  if mode == 'train':
    raise ValueError("Fairblocks Real images are for test purposes only!")
  elif mode == 'eval':
    raise ValueError("Fairblocks Real images are for test purposes only!")
  elif mode == 'test':
    meta_list_file = os.path.join(split_dir, 'test.txt')
  else:
    raise ValueError("Mode %s is not supported!" % mode)
  with open(meta_list_file) as f:
    meta_list = f.read().split('\n')
    meta_list.pop() # remove trailing empty line

  filenames = []
  labels = []
  for i, meta in enumerate(meta_list):
    if (i+1) % 100 == 0:
      print("%s / %s : %s" % (i+1, len(meta_list), meta))
    rec = meta.split(' ')
    filenames.append(os.path.join(data_dir, 'recordings', rec[0]))
    labels.append(float(rec[1]))

  return filenames, labels

def _create_dataset(filenames, labels):
  """
  Creates a dataset from the given filename and label tensors.
  """
  tf_filenames = tf.constant(filenames)
  tf_labels = tf.constant(labels)
  dataset = tf.data.Dataset.from_tensor_slices((tf_filenames, tf_labels))
  return dataset

def _parse_record(filename, label, augment=[]):
  """
  Reads the file and returns a (feature, label) pair.
  """
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string, channels=_CHANNELS)
  image_resized = tf.image.resize_image_with_crop_or_pad(
      image_decoded, _HEIGHT, _WIDTH)
  image_float = tf.cast(image_resized, tf.float32)
  image_float = tf.reshape(image_float, [_HEIGHT, _WIDTH, _CHANNELS])

  return image_float, label

def _center_data(feature, label, rgb_mean):
  """
  Subtracts the mean of the respective data split part to center the data.
  """
  feature_centered = feature - tf.reshape(tf.constant(rgb_mean), [1, 1, 3])
  return feature_centered, label


# public input_fn for dataset iteration

def fairblocks_real_input_fn(
    mode, data_dir, split_name,
    batch_size, num_epochs=1,
    n_prefetch=2, augment=[]):
  """
  Input_fn to feed a tf.estimator.Estimator with images from the real FAIR
  block towers test set.

  Args:
    mode: only 'test' mode is supported
    data_dir: the root directory of the fairblocks_real dataset
    split_name: directory name under data_dir/ containing train.txt, eval.txt and test.txt
    batch_size:
    num_epochs:
    n_prefetch: number of images to prefetch into RAM
    augment: data augmentations to apply
      'subtract_mean': subtracts the RGB mean of the data chunk loaded
  """
  split_dir = os.path.join(data_dir, 'splits', split_name)
  filenames, labels = _get_filenames_with_labels(mode, data_dir, split_dir)
  rgb_mean_npy = np.load(
      os.path.join(split_dir, mode + '_bgr_mean.npy'))[[2, 1, 0]]
  dataset = _create_dataset(filenames, labels)

  # parse data from files and apply pre-processing
  dataset = dataset.map(lambda feature, label: _parse_record(feature, label, augment))

  if 'subtract_mean' in augment:
    dataset = dataset.map(
        lambda feature, label: _center_data(feature, label, rgb_mean_npy))

  # prepare batch and epoch cycle
  dataset = dataset.prefetch(n_prefetch * batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  # set up iterator
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels
