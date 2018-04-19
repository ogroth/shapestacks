"""
Train an InceptionV4 based stability predictor with logistic regression on a
ShapeStack dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import argparse
import pickle
import subprocess
import time
import tensorflow as tf

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from tf_models.inception.inception_model import inception_v4_logregr_model_fn
from data_provider.shapestacks_provider import shapestacks_input_fn
from data_provider.fairblocks_provider import fairblocks_real_input_fn


# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Train an InceptionV4 based stability predictor.')
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
# data augmentation parameters
ARGPARSER.add_argument(
    '--augment', type=str, nargs='+',
    default=['crop', 'recolour', 'flip', 'clip', 'rotate', 'noise', 'stretch'],
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

def main(unparsed_argv):
  """
  Pseudo-main executed via tf.app.run().
  """
  # using the Winograd non-fused algorithms provides a small performance boost
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # set up a RunConfig and the estimator
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=FLAGS.memcap
  )
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  run_config = tf.estimator.RunConfig(
      session_config=sess_config,
      save_checkpoints_secs=1e9 #TODO: make parameter
  )
  classifier = tf.estimator.Estimator(
      model_fn=inception_v4_logregr_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params={'num_display_images' : FLAGS.display_inputs}
      )

  # keeping track of top-n models
  snapshots_dir = os.path.join(FLAGS.model_dir, 'snapshots')
  topn_eval_models_file = os.path.join(snapshots_dir, 'topn_eval_models.dict')
  topn_real_models_file = os.path.join(snapshots_dir, 'topn_real_models.dict')
  if not os.path.exists(topn_eval_models_file):
    topn_eval_models = {}
  else:
    with open(topn_eval_models_file, 'rb') as f:
      topn_eval_models = pickle.load(f)
  if not os.path.exists(topn_real_models_file):
    topn_real_models = {}
  else:
    with open(topn_real_models_file, 'rb') as f:
      topn_real_models = pickle.load(f)

  # main loop
  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):

    # create logging hooks
    tensors_to_log = {
        'logits' : 'logits'
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000) #TODO: make parameter

    # training
    classifier.train(
        input_fn=lambda: shapestacks_input_fn(
            'train', FLAGS.data_dir, FLAGS.split_name,
            FLAGS.batch_size, FLAGS.epochs_per_eval,
            FLAGS.n_prefetch, FLAGS.augment),
        hooks=[logging_hook],)

    # evaluate the model on the corresponding eval set
    eval_results = classifier.evaluate(
        input_fn=lambda: shapestacks_input_fn(
            'eval', FLAGS.data_dir, FLAGS.split_name,
            FLAGS.batch_size, FLAGS.epochs_per_eval,
            FLAGS.n_prefetch, FLAGS.augment),
        name='eval')

    # save model snapshot if within top-N
    last_ckpt_name = os.path.basename(classifier.latest_checkpoint())
    eval_acc = eval_results['accuracy']
    ckpt_name = "eval=%.6f" % (eval_acc, )
    ckpt_dir = os.path.join(FLAGS.model_dir, 'snapshots', ckpt_name)
    if not os.path.exists(snapshots_dir):
      os.mkdir(snapshots_dir)
    if os.path.exists(ckpt_dir):
      shutil.rmtree(ckpt_dir)
    os.mkdir(ckpt_dir)
    for cf in filter(lambda f: f.startswith(last_ckpt_name), os.listdir(FLAGS.model_dir)):
      shutil.copy(os.path.join(FLAGS.model_dir, cf), ckpt_dir)
    with open(os.path.join(ckpt_dir, 'checkpoint'), 'w') as f:
      f.write("model_checkpoint_path: \"%s\"\n" % last_ckpt_name)
    topn_eval_models.update({eval_acc : ckpt_name})
    if len(topn_eval_models) > FLAGS.n_best_eval:
      worst_acc = min(topn_eval_models.keys())
      worst_ckpt_dir = os.path.join(snapshots_dir, topn_eval_models[worst_acc])
      shutil.rmtree(worst_ckpt_dir)
      topn_eval_models.pop(worst_acc)
    with open(topn_eval_models_file, 'wb') as f:
      pickle.dump(topn_eval_models, f)

    # evaluate the model on the real data
    if FLAGS.real_data_dir != '':
      real_results = classifier.evaluate(
          input_fn=lambda: fairblocks_real_input_fn(
              'test',
              FLAGS.real_data_dir,
              'default',
              FLAGS.batch_size, FLAGS.epochs_per_eval,
              FLAGS.n_prefetch, FLAGS.augment),
          name='real')
      # save model snapshot if within top-N
      last_ckpt_name = os.path.basename(classifier.latest_checkpoint())
      real_acc = real_results['accuracy']
      ckpt_name = "real=%.6f" % (real_acc, )
      ckpt_dir = os.path.join(FLAGS.model_dir, 'snapshots', ckpt_name)
      if not os.path.exists(snapshots_dir):
        os.mkdir(snapshots_dir)
      if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
      os.mkdir(ckpt_dir)
      for cf in filter(lambda f: f.startswith(last_ckpt_name), os.listdir(FLAGS.model_dir)):
        shutil.copy(os.path.join(FLAGS.model_dir, cf), ckpt_dir)
      with open(os.path.join(ckpt_dir, 'checkpoint'), 'w') as f:
        f.write("model_checkpoint_path: \"%s\"\n" % last_ckpt_name)
      topn_real_models.update({real_acc : ckpt_name})
      if len(topn_real_models) > FLAGS.n_best_eval:
        worst_acc = min(topn_real_models.keys())
        worst_ckpt_dir = os.path.join(snapshots_dir, topn_real_models[worst_acc])
        shutil.rmtree(worst_ckpt_dir)
        topn_real_models.pop(worst_acc)
      with open(topn_real_models_file, 'wb') as f:
        pickle.dump(topn_real_models, f)


if __name__ == '__main__':
  print("Training an InceptionV4 based logistic regression.")
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)

  # writing arguments and git hash to info file for run identification
  os.makedirs(FLAGS.model_dir, exist_ok=True)
  RUNFILE_FN = "info_"+time.strftime("%m%d_%H%M%S")+".txt"
  RUNFILE_PATH = os.path.join(FLAGS.model_dir, RUNFILE_FN)
  with open(RUNFILE_PATH, "w") as f:
    label = subprocess.check_output(["git", "describe", "--always"]).strip()
    f.write('latest git commit on this branch: '+str(label)+'\n')
    f.write('\nFLAGS: \n')
    for key in vars(FLAGS):
      f.write(key + ': ' + str(vars(FLAGS)[key])+ '\n')
    f.write("\nUNPARSED_ARGV:\n" + str(UNPARSED_ARGV))

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(argv=[sys.argv[0]] + UNPARSED_ARGV)
