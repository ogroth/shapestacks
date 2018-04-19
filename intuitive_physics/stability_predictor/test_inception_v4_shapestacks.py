"""
Test an InceptionV4 stability predictor on the ShapeStacks test split and on
real image data from the FAIR block tower test set.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import argparse
import time
import tensorflow as tf

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from tf_models.inception.inception_model import inception_v4_logregr_model_fn
from data_provider.shapestacks_provider import shapestacks_input_fn
from data_provider.fairblocks_provider import fairblocks_real_input_fn


# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Compute accuracy of an InceptionV4 based stability predictor.')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/datasets/shapestacks',
    help='The path to the data directory.')
ARGPARSER.add_argument(
    '--split_name', type=str, default='ccs_all',
    help="The name of the split to be used.")
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/stability_predictor',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--real_data_dir', type=str, default='',
    help='The path to the real FAIR block tower test set.')
# model parameters
ARGPARSER.add_argument(
    '--display_inputs', type=int, default=0,
    help='The number of input images to display in tensorboard per batch.')
# ARGPARSER.add_argument(
#     '--tfckpt_dir', type=str,
#     help="The directory of the TF model snapshot to use.")
# data augmentation parameters
ARGPARSER.add_argument(
    '--augment', type=str, nargs='+',
    default=['no_augment'],
    help="Apply ImageNet-like training data augmentation.")
# training parameters
ARGPARSER.add_argument(
    '--train_epochs', type=int, default=40,
    help='The number of epochs to train.')
ARGPARSER.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of epochs to run in between evaluations.')
ARGPARSER.add_argument(
    '--batch_size', type=int, default=32,
    help='The number of images per batch.')
ARGPARSER.add_argument(
    '--n_best_eval', type=int, default=5,
    help='Top-N best performing snapshots to keep (according to performance on \
    validation set).')
# memory management parameters
ARGPARSER.add_argument(
    '--memcap', type=float, default=0.8,
    help='Maximum fraction of memory to allocate per GPU.')
ARGPARSER.add_argument(
    '--n_prefetch', type=str, default=32,
    help='How many batches to prefetch into RAM.')


def analyse_checkpoint(dir_snapshot, name_snapshot, unparsed_argv):
  """
  TODO: docstring
  """
  # copy files to FLAGS.model_dir/test_scores
  target_dir = os.path.join(FLAGS.model_dir, 'test_scores_' + name_snapshot)
  print('remove directory ' + target_dir)
  # ignore_errors=1 for dealing with non-existing dir
  shutil.rmtree(target_dir, ignore_errors=1)
  print('copy from directory ' + dir_snapshot)
  print('to directory ' + target_dir)
  shutil.copytree(dir_snapshot, target_dir)

  # set up a RunConfig and the estimator
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=FLAGS.memcap
  )
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  run_config = tf.estimator.RunConfig(
      session_config=sess_config,
  )
  classifier = tf.estimator.Estimator(
      model_fn=inception_v4_logregr_model_fn,
      model_dir=target_dir,
      config=run_config,
      params={'num_display_images' : FLAGS.display_inputs}
      )

  # evaluate the model on the corresponding test set
  test_results = classifier.evaluate(
      input_fn=lambda: shapestacks_input_fn(
          'eval', FLAGS.data_dir, FLAGS.split_name,
          FLAGS.batch_size, FLAGS.epochs_per_eval,
          FLAGS.n_prefetch, FLAGS.augment),
      name='test')
  print(test_results['accuracy'])

  # evaluate the model on real data
  if FLAGS.real_data_dir != '':
    real_results = classifier.evaluate(
        input_fn=lambda: fairblocks_real_input_fn(
            'test',
            FLAGS.real_data_dir,
            'default',
            FLAGS.batch_size, FLAGS.epochs_per_eval,
            FLAGS.n_prefetch, FLAGS.augment),
        name='real')
    print(real_results['accuracy'])

  # writing accuracies and flags to file
  filename = "info_" + time.strftime("%m%d_%H%M%S") + ".txt"
  filename = os.path.join(target_dir, filename)
  with open(filename, "w") as f:
    f.write('Test Accuracy %.3f: \n' %(test_results['accuracy']))
    f.write('\nAccuracy Real Data %.3f: \n' %(real_results['accuracy']))
    f.write('\nFLAGS: \n')
    for key in vars(FLAGS):
      f.write(key + ': ' + str(vars(FLAGS)[key]) + '\n')
    f.write("\nUNPARSED_ARGV:\n" + str(UNPARSED_ARGV))
    f.close()

def main(unparsed_argv):
  """
  Pseudo-main executed via tf.app.run().
  """
  # using the Winograd non-fused algorithms provides a small performance boost
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  snapshot_types = ['real', 'eval']
  for snapshot_type in snapshot_types:
    search_dir = os.path.join(FLAGS.model_dir, 'snapshots')
    list_subdir = [name for name in os.listdir(search_dir) \
        if os.path.isdir(os.path.join(search_dir, name)) \
        and snapshot_type in name]
    list_subdir = [x for _, x in sorted(zip([float(name[5:]) \
        for name in list_subdir], list_subdir), reverse=True)]
    for i in list_subdir:
      print(i + '\n')
    name_snapshot = list_subdir[0]
    dir_snapshot = os.path.join(FLAGS.model_dir, 'snapshots', name_snapshot)
    analyse_checkpoint(dir_snapshot, name_snapshot, unparsed_argv)


if __name__ == '__main__':
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)
  tf.app.run(argv=[sys.argv[0]] + UNPARSED_ARGV)
