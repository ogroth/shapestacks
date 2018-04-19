"""
Contains the model definition for Inception (aka GoogleNet).

Provides functions to create an Inception as a pure computational graph or as a
pre-packaged model function returning a tf.estimator.EstimatorSpec for
convenient training and evaluation via the tf.estimator.Estimator interface.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .inception_v3 import inception_v3
from .inception_v4 import inception_v4


# image defaults
_CHANNELS = 3
_HEIGHT = 224
_WIDTH = 224

# tensorboard defaults
_NUM_DISPLAY_IMAGES = 0

# classification defaults
_NUM_CLASSES = 1000

# learning deafults
_START_LEARNING_RATE = 0.045
_DECAY = 0.9
_EPSILON = 1.0
_LR_DECAY = 0.94
_LR_DECAY_RATE = 2 # decay ever n epochs
_MOMENTUM = 0.9


def inception_v3_model_fn(features, labels, mode, params):
  """
  Sets up the model for training and evaluation phases.
  Adds tensorboard logging.
  Returns a tf.estimator.EstimatorSpec object.

  Args:
    features: a tensor of shape [batch_size, height, width, 3] (NHWC)
    labels: class numbers
    mode:
    params:
      num_classes
      num_display_images
  """
  # parameter parsing or defaults
  num_classes = (
      _NUM_CLASSES if not 'num_classes' in params
      else params['num_classes'])
  num_display_images = (
      _NUM_DISPLAY_IMAGES if not 'num_display_images' in params
      else params['num_display_images'])

  if num_display_images > 0:
    tf.summary.image(
        'images',
        features,
        max_outputs=num_display_images)

  # construct graph
  logits, endpoints = inception_v3(
      inputs=features,
      num_classes=num_classes,
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  for _, endpoint in endpoints.items():
    tf.add_to_collection('inception_v3_endpoints', endpoint)

  # predictions to make
  predictions = {
      'classes' : tf.argmax(logits, axis=1, name='classes'),
      'softmax' : tf.nn.softmax(logits, name='softmax'),
      'logits' : tf.identity(logits, name='logits')
  }

  # loss function to optimize
  if mode != tf.estimator.ModeKeys.PREDICT:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    # evaluation metrics to monitor
    eval_metric_ops = {
        'accuracy' : tf.metrics.accuracy(
            labels=tf.argmax(onehot_labels, axis=1),
            predictions=predictions['classes'])
    }

  # return EstimatorSpec depending on mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=_START_LEARNING_RATE,
        decay=_DECAY,
        epsilon=_EPSILON)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

def inception_v3_logregr_model_fn(features, labels, mode, params):
  """
  Sets up an InceptionV3 based logistic regression model for training and
  evaluation phases.
  Adds tensorboard logging.
  Returns a tf.estimator.EstimatorSpec object.

  Args:
    features: a tensor of shape [batch_size, height, width, 3] (NHWC)
    labels: class numbers
    mode:
    params:
      num_display_images
  """
  # parameter parsing or defaults
  num_classes = 1
  num_display_images = (
      _NUM_DISPLAY_IMAGES if not 'num_display_images' in params
      else params['num_display_images'])

  if num_display_images > 0:
    tf.summary.image(
        'images',
        features,
        max_outputs=num_display_images)

  # construct graph
  logits, endpoints = inception_v3(
      inputs=features,
      num_classes=num_classes,
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  for _, endpoint in endpoints.items():
    tf.add_to_collection('inception_v3_endpoints', endpoint)

  # predictions to make
  log_regr = tf.nn.sigmoid(logits, 'sigmoid')
  predictions = {
      'classes' : tf.round(log_regr, name='classes'),
      'probabilities' : log_regr,
      'logits' : tf.identity(logits, name='logits')
  }

  # loss function to optimize
  if mode != tf.estimator.ModeKeys.PREDICT:
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.reshape(tf.cast(labels, tf.int32), [-1, 1]), logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    # evaluation metrics to monitor
    eval_metric_ops = {
        'accuracy' : tf.metrics.accuracy(
            labels=tf.reshape(tf.cast(labels, tf.int32), [-1, 1]),
            predictions=predictions['classes'])
    }

  # return EstimatorSpec depending on mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=_START_LEARNING_RATE,
        decay=_DECAY,
        epsilon=_EPSILON)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

def inception_v4_model_fn(features, labels, mode, params):
  """
  Sets up the model for training and evaluation phases.
  Adds tensorboard logging.
  Returns a tf.estimator.EstimatorSpec object.

  Args:
    features: a tensor of shape [batch_size, height, width, 3] (NHWC)
    labels: class numbers
    mode:
    params:
      num_classes
      num_display_images
  """
  # parameter parsing or defaults
  num_classes = (
      _NUM_CLASSES if not 'num_classes' in params
      else params['num_classes'])
  num_display_images = (
      _NUM_DISPLAY_IMAGES if not 'num_display_images' in params
      else params['num_display_images'])

  if num_display_images > 0:
    tf.summary.image(
        'images',
        features,
        max_outputs=num_display_images)

  # construct graph
  logits, endpoints = inception_v4(
      inputs=features,
      num_classes=num_classes,
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  for _, endpoint in endpoints.items():
    tf.add_to_collection('inception_v4_endpoints', endpoint)

  # predictions to make
  predictions = {
      'classes' : tf.argmax(logits, axis=1, name='classes'),
      'softmax' : tf.nn.softmax(logits, name='softmax'),
      'logits' : tf.identity(logits, name='logits')
  }

  # loss function to optimize
  if mode != tf.estimator.ModeKeys.PREDICT:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    # evaluation metrics to monitor
    eval_metric_ops = {
        'accuracy' : tf.metrics.accuracy(
            labels=tf.argmax(onehot_labels, axis=1),
            predictions=predictions['classes'])
    }

  # return EstimatorSpec depending on mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=_START_LEARNING_RATE,
        decay=_DECAY,
        epsilon=_EPSILON)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

def inception_v4_logregr_model_fn(features, labels, mode, params):
  """
  Sets up an InceptionV4 based logistic regression model for training and
  evaluation phases.
  Adds tensorboard logging.
  Returns a tf.estimator.EstimatorSpec object.

  Args:
    features: a tensor of shape [batch_size, height, width, 3] (NHWC)
    labels: class numbers
    mode:
    params:
      num_display_images
  """
  # parameter parsing or defaults
  num_classes = 1
  num_display_images = (
      _NUM_DISPLAY_IMAGES if not 'num_display_images' in params
      else params['num_display_images'])

  if num_display_images > 0:
    tf.summary.image(
        'images',
        features,
        max_outputs=num_display_images)

  # construct graph
  logits, endpoints = inception_v4(
      inputs=features,
      num_classes=num_classes,
      is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  for _, endpoint in endpoints.items():
    tf.add_to_collection('inception_v4_endpoints', endpoint)

  # predictions to make
  log_regr = tf.nn.sigmoid(logits, name='sigmoid')
  predictions = {
      'classes' : tf.round(log_regr, name='classes'),
      'probabilities' : log_regr,
      'logits' : tf.identity(logits, name='logits')
  }

  # loss function to optimize
  if mode != tf.estimator.ModeKeys.PREDICT:
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.reshape(tf.cast(labels, tf.int32), [-1, 1]), logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    # evaluation metrics to monitor
    eval_metric_ops = {
        'accuracy' : tf.metrics.accuracy(
            labels=tf.reshape(tf.cast(labels, tf.int32), [-1, 1]),
            predictions=predictions['classes'])
    }

  # return EstimatorSpec depending on mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=_START_LEARNING_RATE,
        decay=_DECAY,
        epsilon=_EPSILON)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)
