"""
Stacking simulator for cuboids with simulated annealing position sampling at
fixed height.

On Ubuntu 16.04 execute with for offscreen rendering:
LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so
"""

import sys
import os
import argparse
import re
import math
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf

import mujoco_py
from mujoco_py.modder import CameraModder
from mujoco_py.generated.const import FB_OFFSCREEN, FB_WINDOW

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from tf_models.inception.inception_v4 import inception_v4
from utilities.rotation_utils import quaternion_from_euler, euler_from_quaternion
from utilities.mujoco_utils import mjsim_mat_id2name, mjhlp_geom_type_id2name

# command line arguments
ARGPARSER = argparse.ArgumentParser(
    description='Steer a stacking scenario in MuJoCo with a TF stability \
    estimator.')
# directory setup
ARGPARSER.add_argument(
    '--mjmodel_path', type=str,
    help="The path to the XML definition of the MuJoCo model.")
ARGPARSER.add_argument(
    '--tfmodel', type=str, default='inception_v4',
    help="The stability predictor architecture to use. \
    Available: inception_v4")
ARGPARSER.add_argument(
    '--tfckpt_dir', type=str,
    help="The directory of the TF model snapshot to use.")
ARGPARSER.add_argument(
    '--mjsim_dir', type=str,
    help="The directory to log the simulation outcomes to. Will be created if \
    not present, yet.")
ARGPARSER.add_argument(
    '--run_prfx', type=str, default='r1',
    help="The prefix for the log files in mjsim_dir.")
# stacking mode
ARGPARSER.add_argument(
    '--mode', type=str, default='stacking',
    help="Run mode of stacker. Available: stacking | ranking | balancing.")
# camera parameters
ARGPARSER.add_argument(
    '--cameras', type=str, nargs='+',
    default=['cam_1', 'cam_4', 'cam_7', 'cam_10', 'cam_13', 'cam_15'],
    help="The cameras to observe the scene with (cam_1 through cam_16).")
# simulation parameters
ARGPARSER.add_argument(
    '--velocity_thres', type=float, default=0.2,
    help="The maximum velocity allowed before considered a collapse.")
ARGPARSER.add_argument(
    '--placement_steps', type=int, default=50,
    help="The number of simulation steps to let a dropped object settle.")
# simulated annealing
ARGPARSER.add_argument(
    '--stability_thres', type=float, default=0.5,
    help="The maximum score allowed for a position to be considered stable.")
ARGPARSER.add_argument(
    '--anneal_steps', type=int, default=100,
    help="Number of simulation steps before temperature is decreased.")
ARGPARSER.add_argument(
    '--anneal_scale', type=float, default=0.95,
    help="The scale by which the temperature is reduced.")
ARGPARSER.add_argument(
    '--max_sa_steps', type=int, default=500,
    help="The maximum number of simulated annealing steps.")
ARGPARSER.add_argument(
    '--max_move_xy', type=float, default=5.0,
    help="The maximum movement allowed in X and Y directions.")
ARGPARSER.add_argument(
    '--max_pos_xy', type=float, default=2.5,
    help="The maximum coordinates allowed in X and Y directions.")
# rendering parameters
ARGPARSER.add_argument(
    '--rendering_mode', type=str, default="offscreen",
    help="Show simulation in onscreen window or compute offscreen without GUI. \
    Available modes: onscreen | offscreen.")
ARGPARSER.add_argument(
    '--random_colors', action='store_true')
ARGPARSER.add_argument(
    '--random_lights', action='store_true')
ARGPARSER.add_argument(
    '--random_textures', action='store_true')
# debugging
ARGPARSER.add_argument(
    '--debug', action='store_true')

# rendering constants
RENDER_HEIGHT = 224
RENDER_WIDTH = 224
RENDER_CHANNELS = 3

# camera constants
INIT_CAM_ADJUSTMENT = -3.0

# simulation constants
STEPS_STACKING_ROUND = 1000 # amount of simulation steps allowed for one stacking step
STEPS_PLACEMENT = 100 # amount of simulation steps allowed for object to settle
VELOCITY_THRES = 0.5 # max. velocity before being considered a collapse

# predictor constants
CONF_INTERVAL = 1e-5 # positions within this interval are averaged

# object constants
OBJ_FLOAT_Z_OFFSET = 0.1
OBJ_COLORS_RGBA = [
    [1, 0, 0, 1],  # red
    [0, 1, 0, 1],  # green
    [0, 0, 1, 1],  # blue
    [1, 1, 0, 1],  # yellow
    [0, 1, 1, 1],  # cyan
    [1, 0, 1, 1],  # magenta
]

# simulated annealing constants
INIT_TEMP = 1.0


# simulation functions

def parse_object_names(mjmodel):
  """
  Identifies all objects to stack and returns a list of their names.
  """
  return [g for g in list(mjmodel.geom_names) if g.startswith('obj_')]

def _init_objects_stacking(obj_list):
  """
  Initializes the objects for the stacking challenge.
  Modifies MJMODEL.
  """
  for obj_name in obj_list:
    mj_obj_id = MJMODEL.geom_name2id(obj_name)
    obj_color = MJMODEL.geom_rgba[mj_obj_id]
    obj_color[3] = 0.0 # set alpha channel 0.0 to make invisible
    MJMODEL.geom_rgba[mj_obj_id] = obj_color

def _init_cameras(cam_list):
  """
  Initializes the camera positions.
  Modifies MJMODEL.
  """
  # initialize cameras
  for cam in cam_list:
    cx, cy, cz = CAM_MODDER.get_pos(cam)
    CAM_MODDER.set_pos(cam, (cx, cy, cz + INIT_CAM_ADJUSTMENT))


# TF functions

def _init_stability_predictor(tfmodel, tfckpt_dir):
  """
  Loads a stability predictor from the checkpoint directory.
  Returns an initialized session and the in- and output endpoints of the
  predictor.
  """
  input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
  inputs = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')
  num_classes = 1
  if tfmodel == 'inception_v4':
    logits, endpoints = inception_v4(
        inputs=inputs,
        num_classes=num_classes,
        is_training=False)
  else:
    raise Exception("Unsupported model architecture %s!" % tfmodel)
  log_regr = tf.nn.sigmoid(logits, 'sigmoid')
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=0.8
  )
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  sess = tf.Session(config=sess_config)
  saver = tf.train.Saver()
  ckpt_path = tf.train.latest_checkpoint(tfckpt_dir)
  saver.restore(sess, ckpt_path)
  model_endpoints = {'inputs' : inputs, 'scores' : log_regr}
  return sess, model_endpoints


# object manipulation

def _activate_object(obj_name: str):
  """
  Makes the selected object visible.
  Modifies MJMODEL!
  """
  mj_obj_id = MJMODEL.geom_name2id(obj_name)
  obj_color = MJMODEL.geom_rgba[mj_obj_id]
  obj_color[3] = 1.0 # set alpha channel to make visible
  MJMODEL.geom_rgba[mj_obj_id] = obj_color

def _deactivate_object(obj_name: str):
  """
  Makes the selected object invisible.
  Modifies MJMODEL!
  """
  mj_obj_id = MJMODEL.geom_name2id(obj_name)
  obj_color = MJMODEL.geom_rgba[mj_obj_id]
  obj_color[3] = 0.0 # set alpha channel to make invisible
  MJMODEL.geom_rgba[mj_obj_id] = obj_color

def _get_object_shape(obj_name: str):
  """
  Looks up the object's shape in the model definition.
  """
  mj_obj_id = MJMODEL.geom_name2id(obj_name)
  obj_shape = mjhlp_geom_type_id2name(MJMODEL.geom_type[mj_obj_id])
  return obj_shape

def _get_object_size(obj_name: str):
  """
  Gets the object size.
  """
  mj_obj_id = MJMODEL.geom_name2id(obj_name)
  obj_size = MJMODEL.geom_size[mj_obj_id]
  return obj_size

def _get_object_pos(obj_name: str):
  """
  Gets the current position of an object in absolute X-Y-Z world coordinates.
  """
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  obj_qpos = state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0]+3]
  return obj_qpos

def _set_object_pos(obj_name: str, new_pos):
  """
  Moves the object of the specified name to the given position (in absolute
  X-Y-Z world coordinates).
  Modifies the state of MJSIM!

  params:
    - obj_name: str
    - new_pos: [x, y, z]
  """
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  state.qpos[obj_qpos_addr[0]:obj_qpos_addr[0]+3] = np.array(new_pos)
  MJSIM.set_state(state)

def _get_object_quat(obj_name: str):
  """
  Gets the current quaternion of the specified object.
  """
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  obj_quat = state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7]
  return obj_quat

def _set_object_quat(obj_name: str, quat):
  """
  Rotates the object by given quaternion.
  Modifies the state of MJSIM!

  params:
    - quaternion: [w, x, y, z]
  """
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7] = quat
  MJSIM.set_state(state)

def _set_object_euler(obj_name: str, euler):
  """
  Sets the object's orientation to the given euler angle.
  Modifies the state of MJSIM!

  params:
    - euler: [x, y, z] (in degrees)
  """
  euler = np.array(euler)
  euler *= (2.0 * np.pi / 360.0)
  quat = quaternion_from_euler(
      euler[0], euler[1], euler[2],
      axes='sxyz')
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7] = quat
  MJSIM.set_state(state)

def _rotate_object(obj_name: str, euler):
  """
  Rotates the specified object by the given euler angle!
  Modifies the state of MJSIM!

  params:
    - euler: [x, y, z] (in degrees)
  """
  # convert euler to radians
  euler = np.array(euler)
  euler *= (2.0 * np.pi / 360.0)
  # read current orientation as euler and add rotation
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  obj_quat = state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7]
  obj_euler = euler_from_quaternion(obj_quat, axes='sxyz')
  obj_euler += euler
  # set orientation as quaternion
  quat = quaternion_from_euler(
      obj_euler[0], obj_euler[1], obj_euler[2],
      axes='sxyz')
  state.qpos[obj_qpos_addr[0]+3:obj_qpos_addr[0]+7] = quat
  MJSIM.set_state(state)

def _get_object_qpos(obj_name: str):
  """
  Returns the object's qpos (xyz + quat).
  """
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  obj_qpos = state.qpos[obj_qpos_addr[0]:obj_qpos_addr[1]]
  return obj_qpos

def _set_object_qpos(obj_name: str, qpos):
  """
  Rotates the object by given quaternion.
  Modifies the state of MJSIM!

  params:
    - qpos: xyz + quat
  """
  state = MJSIM.get_state()
  obj_qpos_addr = MJSIM.model.get_joint_qpos_addr(obj_name)
  state.qpos[obj_qpos_addr[0]:obj_qpos_addr[1]] = qpos
  MJSIM.set_state(state)

def _get_object_vel(obj_name: str):
  """
  Gets the current object velocity.
  """
  state = MJSIM.get_state()
  obj_qvel_addr = MJSIM.model.get_joint_qvel_addr(obj_name)
  obj_qvel = state.qvel[obj_qvel_addr[0]:obj_qvel_addr[1]]
  return obj_qvel

def _set_object_vel(obj_name: str, vel):
  """
  Sets the velocity values for the specified object.
  Modifies the state of MJSIM!

  params:
    - vel: [fb, ud, lr, y, p, r]
  """
  state = MJSIM.get_state()
  obj_qvel_addr = MJSIM.model.get_joint_qvel_addr(obj_name)
  state.qvel[obj_qvel_addr[0]:obj_qvel_addr[1]] = vel
  MJSIM.set_state(state)


# simulated annealing functions

def _sa_next_pos(cur_x, cur_y):
  """
  Samples the next position from the current one according to a normal
  distribution, limited by a maximum move range and the scene boundaries.
  """
  while True:
    off_x = np.clip(np.random.normal(), -MAX_MOVE_XY, MAX_MOVE_XY)
    off_y = np.clip(np.random.normal(), -MAX_MOVE_XY, MAX_MOVE_XY)
    new_x = np.clip(cur_x + off_x, -MAX_POS_XY, MAX_POS_XY)
    new_y = np.clip(cur_y + off_y, -MAX_POS_XY, MAX_POS_XY)
    if new_y ** 2 + new_y **2 <= MAX_POS_XY ** 2: # new position within circle
      break
  return new_x, new_y

def _sa_acceptance_prob(c0, c1, T):
  """
  params:
    - c0: previous cost
    - c1: new cost
    - T: current temperature
  """
  return np.exp(-(c1 - c0) / T)


# scene setup

def _init_scene():
  """
  Initializes the scene.
  Randomizes the object colors, background textures and light conditions.
  """
  if FLAGS.random_colors: # randomize object colors
    for mj_geom_name in \
      filter(lambda n: n.startswith('shape_') or n.startswith('obj_'), \
        MJMODEL.geom_names):
      mj_geom_id = MJMODEL.geom_name2id(mj_geom_name)
      # mj_geom_rgba[3] = 1.0
      mj_geom_rgba = OBJ_COLORS_RGBA[np.random.randint(0, len(OBJ_COLORS_RGBA))]
      MJMODEL.geom_rgba[mj_geom_id] = mj_geom_rgba

  if FLAGS.random_lights: # set main light to cast shadow
    lm = mujoco_py.modder.LightModder(MJSIM)
    mj_light_names = MJMODEL.light_names
    mj_light_id = np.random.randint(0, len(mj_light_names))
    mj_light_name = mj_light_names[mj_light_id]
    lm.set_castshadow(mj_light_name, 1)

  if FLAGS.random_textures: # randomize textures according to asset catalog
    # find materials
    mat_id2name = mjsim_mat_id2name(MJSIM)
    mat_floor_id2name = dict(
        filter(lambda t: t[1].startswith('mat_floor_'), mat_id2name.items()))
    mat_wall_id2name = dict(
        filter(lambda t: t[1].startswith('mat_wall_'), mat_id2name.items()))
    rnd_floortex = np.random.randint(0, len(mat_floor_id2name))
    rnd_walltex = np.random.randint(0, len(mat_wall_id2name))
    matid_floortex = sorted(mat_floor_id2name.items(), key=lambda t: t[0])[rnd_floortex][0]
    matid_walltex = sorted(mat_wall_id2name.items(), key=lambda t: t[0])[rnd_walltex][0]
    # set wall and floor materials
    for plane, matid in \
      [('floor', matid_floortex), ('wall_1', matid_walltex), ('wall_2', matid_walltex)]:
      geom_id = MJMODEL.geom_name2id(plane)
      MJMODEL.geom_matid[geom_id] = matid

  MJSIM.step() # forward simulation to update scene

# simulation modes

def _idle_mode():
  """
  Just lets the simulation idle until it is terminated.
  """
  while True:
    MJSIM.step()
    if FLAGS.rendering_mode == 'onscreen':
      mujoco_py.functions.mjr_setBuffer(FB_WINDOW, MJSIM.render_contexts[0].con)
      MJVIEWER.render()

def _stackability_estimation_mode(obj_list):
  """
  Iterates over object list and computes 'stackability' for each object. The
  object under investigation is placed at the origin in N discrete possible
  orientations (defined by object class). All other objects are then floated
  over the base object and the best stability scores are aggregated per
  orientation.
  """

  print("Estimating stackability of all given objects...")

  # obj_handle = {'name' : None, 'orig_qpos' : None, 'init_qpos' : None}
  base_obj_dict = {} # DEBUG
  obj_pair_queue = [] # (base_obj_handle, mover_obj_handle)

  # pre-compute all estimation setups
  # for base_obj_name in obj_list:
  for base_obj_name in filter(lambda obj_name: '_base_' in obj_name, obj_list):

    # original qpos of base object and size
    base_obj_orig_qpos = _get_object_qpos(base_obj_name)
    base_obj_orig_quat = base_obj_orig_qpos[3:7]
    base_obj_euler_str = re.search(r'euler=\d+_\d+_\d+', base_obj_name).group(0)
    base_obj_euler_str = base_obj_euler_str.lstrip('euler=')
    base_obj_euler = np.array([int(d) for d in base_obj_euler_str.split('_')])
    print("base:", base_obj_name, base_obj_orig_quat, base_obj_euler)

    # get object shape and init possible orientations
    base_obj_size = _get_object_size(base_obj_name)
    base_obj_shape = _get_object_shape(base_obj_name)
    if base_obj_shape == 'sphere':
      base_obj_init_quats = [quaternion_from_euler(0.0, 0.0, 0.0)]
      base_obj_init_heights = [base_obj_size[0] * 2.0]
    elif base_obj_shape == 'cylinder':
      base_obj_init_quats = [
          quaternion_from_euler(0.0, 0.0, 0.0),
          quaternion_from_euler(90.0 / 360.0 * 2 * np.pi, 0.0, 0.0)]
      if base_obj_euler[0] % 180 == 90: # sideways cylinder picked up
        base_obj_init_heights = [
            base_obj_size[0] * 2.0,   # sideways cylinder
            base_obj_size[1] * 2.0]   # upright cylinder
      elif base_obj_euler[0] % 180 == 0: # upright cylinder picked up
        base_obj_init_heights = [
            base_obj_size[1] * 2.0,   # upright cylinder
            base_obj_size[0] * 2.0]   # sideways cylinder
      else:
        raise Exception("Cannot handle %s with initial orientation %s" \
            % (base_obj_shape, base_obj_euler))
    elif base_obj_shape == 'box':
      base_obj_init_quats = [
          quaternion_from_euler(0.0, 0.0, 0.0),
          quaternion_from_euler(90.0 / 360.0 * 2 * np.pi, 0.0, 0.0),
          quaternion_from_euler(0.0, 90.0 / 360.0 * 2 * np.pi, 0.0)]
      base_obj_init_heights = [
          base_obj_size[2] * 2.0,   # Z upright
          base_obj_size[1] * 2.0,   # X upright
          base_obj_size[0] * 2.0]   # Y upright
    else:
      raise Exception("Cannot handle shape type %s!" % base_obj_shape)

    # init all possible mover objects
    mover_list = set(obj_list)
    mover_list = set(list(filter(lambda obj_name: '_mover_' in obj_name, obj_list)))
    mover_list.remove(base_obj_name)
    mover_list = list(mover_list)

    # iterate over all base and mover combinations
    for base_obj_init_quat, base_obj_init_h \
      in zip(base_obj_init_quats, base_obj_init_heights):

      # compute base_obj_init_qpos
      base_obj_init_qpos = np.zeros(7)
      base_obj_init_qpos[2] = base_obj_init_h / 2.0
      base_obj_init_qpos[3:] = np.array(base_obj_init_quat)

      # lookup for base object configurations
      base_obj_handle = {
          'name' : base_obj_name,
          'orig_qpos' : base_obj_orig_qpos,
          'init_qpos' : base_obj_init_qpos}
      base_obj_key = base_obj_name + '#' + str(base_obj_init_qpos)
      base_obj_dict.update({base_obj_key : base_obj_handle})

      # iterate over all possible movers
      for mover_obj_name in mover_list:

        # original qpos of base object and size
        mover_obj_orig_qpos = _get_object_qpos(mover_obj_name)
        mover_obj_orig_quat = mover_obj_orig_qpos[3:7]
        mover_obj_euler_str = re.search(r'euler=\d+_\d+_\d+', mover_obj_name).group(0)
        mover_obj_euler_str = mover_obj_euler_str.lstrip('euler=')
        mover_obj_euler = np.array([int(d) for d in mover_obj_euler_str.split('_')])
        print("mover:", mover_obj_name, mover_obj_orig_quat, mover_obj_euler)

        # get object shape and init possible orientations
        mover_obj_size = _get_object_size(mover_obj_name)
        mover_obj_shape = _get_object_shape(mover_obj_name)
        if mover_obj_shape == 'sphere':
          mover_obj_h = mover_obj_size[0] * 2.0
        elif mover_obj_shape == 'cylinder':
          if mover_obj_euler[0] % 180 == 90: # sideways cylinder picked up
            mover_obj_h = mover_obj_size[0] * 2.0
          elif mover_obj_euler[0] % 180 == 0: # upright cylinder picked up
            mover_obj_h = mover_obj_size[1] * 2.0
          else:
            raise Exception("Cannot handle %s with initial orientation %s" \
                % (mover_obj_shape, mover_obj_euler))
        elif mover_obj_shape == 'box':
          mover_obj_h = mover_obj_size[2] * 2.0
        else:
          raise Exception("Cannot handle shape type %s!" % base_obj_shape)

        mover_obj_init_qpos = np.zeros(7)
        mover_obj_init_qpos[2] = \
          base_obj_init_h + mover_obj_h / 2.0 + OBJ_FLOAT_Z_OFFSET
        mover_obj_init_qpos[3:] = mover_obj_orig_qpos[3:] # set quat
        mover_obj_handle = {
            'name' : mover_obj_name,
            'orig_qpos' : mover_obj_orig_qpos,
            'init_qpos' : mover_obj_init_qpos
        }
        obj_pair_queue.append((base_obj_handle, mover_obj_handle))

  # global counters for stackability
  base_ranking = {}
  # local counters stackability (reset after every stacking trial)
  t = 0
  cur_obj_pair = None
  temp = INIT_TEMP

  while True:

    # exit mode when no object moving or queued any more
    if cur_obj_pair is None \
      and len(obj_pair_queue) == 0:
      # DEBUG: drop objects according to stackability
      base_configurations = sorted(base_ranking.items(), key=lambda t: t[1])
      base_objects = [k.split('#')[0] for k, v in base_configurations]
      placed_objects = set()
      stack_height = 0.0
      for i, (obj_name, (obj_key, stackability_score)) in \
        enumerate(zip(base_objects, base_configurations)):
        # add to stackability ranking
        mj_geom_id = MJMODEL.geom_name2id(obj_name)
        mj_geom_type = mjhlp_geom_type_id2name(MJMODEL.geom_type[mj_geom_id])
        mj_geom_size = MJMODEL.geom_size[mj_geom_id]
        if mj_geom_type == 'box':
          volume = np.prod(mj_geom_size * 2.0)
        elif mj_geom_type == 'cylinder':
          volume = np.pi * mj_geom_size[0] ** 2.0 * (mj_geom_size[1] * 2.0)
        elif mj_geom_type == 'sphere':
          volume = 4.0 / 3.0 * np.pi * mj_geom_size[0] ** 3.0
        else:
          raise Exception('Unable to compute volume for geom type %s!' % mj_geom_type)
        if mj_geom_type == 'box':
          pos = base_obj_dict[obj_key]['init_qpos'][0:3]
          area = volume / (pos[2] * 2.0)
        elif mj_geom_type == 'cylinder':
          quat = base_obj_dict[obj_key]['init_qpos'][3:7]
          if quat[0] == 1.0: # upright
            area = np.pi * mj_geom_size[0] ** 2.0
            mj_geom_type = 'cylinder_upright'
          else: # sideways
            area = (mj_geom_size[0] * 2.0) * (mj_geom_size[1] * 2.0)
            mj_geom_type = 'cylinder_sideways'
        elif mj_geom_type == 'sphere':
          area = np.pi * mj_geom_size[0] ** 2.0
        else:
          raise Exception('Unable to compute volume for geom type %s!' % mj_geom_type)
        print('###')
        print(obj_name)
        print(mj_geom_type)
        print(base_obj_dict[obj_key]['init_qpos'])
        print(volume)
        print(area)
        print(stackability_score)
        STACKABILITY.append((obj_name, mj_geom_type, area, volume, stackability_score))
        # build stacking queue
        if obj_name in placed_objects:
          print("%s already placed in different orientation!" % obj_name)
          continue
        else:
          placed_objects.add(obj_name)
          STACKING_QUEUE.append((obj_name, base_obj_dict[obj_key]['init_qpos'], stackability_score))
      break

    # throw the next available object in
    if cur_obj_pair is None \
      and len(obj_pair_queue) > 0:
      # get next pair
      cur_obj_pair = obj_pair_queue.pop()
      bh, mh = cur_obj_pair
      # spawn base
      _activate_object(bh['name'])
      _set_object_qpos(bh['name'], bh['init_qpos'])
      _set_object_vel(bh['name'], np.zeros(6))
      # spawn mover and init simulated annealing
      _activate_object(mh['name'])
      mover_init_qpos = mh['init_qpos']
      temp = INIT_TEMP
      best_conf, prev_conf = 1.0, 1.0
      score_list, best_pos_list = [], []
      m_prev_x, m_prev_y = 0.0, 0.0
      m_cur_x, m_cur_y = _sa_next_pos(m_prev_x, m_prev_y)
      mover_init_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(mh['name'], mover_init_qpos)
      _set_object_vel(mh['name'], np.zeros(6))
      # DEBUG
      base_obj_key = bh['name'] + '#' + str(bh['init_qpos'])
      mover_obj_key = mh['name'] + '#' + str(mh['init_qpos'])
      print("Estimating stackability for \n base = %s \n mover = %s" % \
          (base_obj_key, mover_obj_key))

    # perform simulated annealing
    if cur_obj_pair \
      and t < MAX_SA_STEPS:
      bh, mh = cur_obj_pair
      # move mover to new position
      mover_qpos = _get_object_qpos(mh['name'])
      m_cur_x, m_cur_y = _sa_next_pos(m_prev_x, m_prev_y)
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(mh['name'], mover_qpos)
      _set_object_vel(mh['name'], np.zeros(6))

      # refresh rendering and evaluate position
      MJSIM.step()
      if FLAGS.rendering_mode == 'onscreen':
        mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
      MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
      frame_list = []
      for cam in CAM_LIST:
        frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
        frame = np.flip(frame, 0)
        frame_list.append(frame)
      frames = np.concatenate(frame_list)
      input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
      feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
      scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
      cur_conf = np.average(scores)

      # accept or reject step based on simulated annealing
      if t % ANNEAL_STEPS: # cool down
        temp *= ANNEAL_SCALE
      a = _sa_acceptance_prob(prev_conf, cur_conf, temp)
      p = np.random.uniform()
      if a > p: # accept the new step
        m_prev_x, m_prev_y = m_cur_x, m_cur_y
        prev_conf = cur_conf
      else: # reject and reset to previous position
        pass

      if cur_conf < best_conf - CONF_INTERVAL: # new best position
        best_conf = cur_conf
        best_pos_list = [(m_cur_x, m_cur_y)]
        print("New best: conf=%s, x=%.4f, y=%.4f" % (best_conf, m_cur_x, m_cur_y))
      elif math.fabs(cur_conf - best_conf) < CONF_INTERVAL: # equally good position
        best_pos_list.append((m_cur_x, m_cur_y))
        print("Additional best: conf=%s, x=%.4f, y=%.4f" % (best_conf, m_cur_x, m_cur_y))

    # put the current object back to its original position
    if cur_obj_pair \
      and t > MAX_SA_STEPS:
      bh, mh = cur_obj_pair
      # perturb best position
      mover_qpos = _get_object_qpos(mh['name'])
      pert_offset = 0.25
      pert_scores = []
      for pert in \
        [
            (-pert_offset, -pert_offset),
            (-pert_offset, pert_offset),
            (pert_offset, -pert_offset),
            (pert_offset, pert_offset)
        ]:
        m_cur_x = np.average([t[0] for t in best_pos_list]) + pert[0]
        m_cur_y = np.average([t[1] for t in best_pos_list]) + pert[1]
        mover_new_qpos = np.copy(mover_qpos)
        mover_new_qpos[0:2] = np.array([m_cur_x, m_cur_y])
        _set_object_qpos(mh['name'], mover_new_qpos)
        _set_object_vel(mh['name'], np.zeros(6))

        MJSIM.step() # refresh rendering and compute scores
        if FLAGS.rendering_mode == 'onscreen':
          mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
        MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
        frame_list = []
        for cam in CAM_LIST:
          frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
          frame = np.flip(frame, 0)
          frame_list.append(frame)
        frames = np.concatenate(frame_list)
        input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
        feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
        scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
        pert_conf = np.average(scores)
        pert_scores.append(pert_conf)
        print("Perturbed best: conf=%s, x=%.4f, y=%.4f" % (pert_conf, m_cur_x, m_cur_y))

      # store the stackability information
      base_obj_key = bh['name'] + '#' + str(bh['init_qpos'])
      if not base_obj_key in base_ranking:
        base_ranking.update({base_obj_key : np.average(pert_scores)})
      else:
        base_ranking[base_obj_key] += np.average(pert_scores)
      # reset base
      _set_object_qpos(bh['name'], bh['orig_qpos'])
      _set_object_vel(bh['name'], np.zeros(6))
      _deactivate_object(bh['name'])
      # reset mover
      _set_object_qpos(mh['name'], mh['orig_qpos'])
      _set_object_vel(mh['name'], np.zeros(6))
      _deactivate_object(mh['name'])
      # reset pair
      t = 0
      cur_obj_pair = None

    # advance simulation after applying all changes
    MJSIM.step()
    t += 1

    # render
    if FLAGS.rendering_mode == 'onscreen':
      mujoco_py.functions.mjr_setBuffer(FB_WINDOW, MJSIM.render_contexts[0].con)
      MJVIEWER.render()

def _stacking_mode(obj_queue):
  """
  Stack all objects in their most suitable orientations according to the global
  STACKING_QUEUE.
  """
  print("Performing stacking...")

  # global counters
  idx_obj = 0
  stack_height = 0.0
  stack_collapsed = False
  # local counters (reset after every stack)
  cur_obj_handle = None
  obj_orig_qpos = None
  t = 0
  temp = INIT_TEMP
  best_conf, prev_conf = 1.0, 1.0
  score_list, best_pos_list = [], []
  m_prev_x, m_prev_y = 0.0, 0.0

  while True:

    # terminate after last object has been placed
    if cur_obj_handle is None \
      and t // (MAX_SA_STEPS + STEPS_PLACEMENT) > 0 \
      and len(obj_queue) == 0: # no more objects to stack
      # check for violation
      velocities = np.abs(MJSIM.data.sensordata)
      stack_collapsed = np.any(velocities > VELOCITY_THRES)
      if stack_collapsed:
        print("COLLAPSE!")
      else:
        print("STABLE!")
      # final report
      print("Done stacking objects!")
      if stack_collapsed:
        idx_obj -= 1
      else:
        pass
      print("Final height: %s" % (idx_obj,))
      break

    # check for stability violations while object is settling
    if cur_obj_handle is None \
      and t // (MAX_SA_STEPS + STEPS_PLACEMENT) > 0 \
      and t % STEPS_STACKING_ROUND != 0:
      velocities = np.abs(MJSIM.data.sensordata)
      stack_collapsed = np.any(velocities > VELOCITY_THRES)
      if stack_collapsed:
        obj_queue.clear()

    # pick up next object
    if cur_obj_handle is None \
      and len(obj_queue) > 0 \
      and t % STEPS_STACKING_ROUND == 0:
      # get next object
      cur_obj_handle = obj_queue.pop(0) # (name, init_qpos, score)
      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      obj_orig_qpos = _get_object_qpos(cur_obj_name)
      _activate_object(cur_obj_name)
      idx_obj += 1
      print("Picked up %s" % (cur_obj_name,))
      # compute spawn position
      obj_h = obj_init_qpos[2] * 2.0
      obj_spawn_qpos = np.copy(obj_init_qpos)
      m_cur_x, m_cur_y = _sa_next_pos(m_prev_x, m_prev_y)
      obj_spawn_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      obj_spawn_qpos[2] += (stack_height + OBJ_FLOAT_Z_OFFSET) # adjust spawn height
      # move object into position
      _set_object_qpos(cur_obj_name, obj_spawn_qpos)
      print("Spawned %s at %s" % (cur_obj_name, obj_spawn_qpos))
      _rotate_object(cur_obj_name, [0.0, 0.0, float(np.random.randint(0, 360))])
      _set_object_vel(cur_obj_name, np.zeros(6))
      # adjust cameras
      for cam in CAM_LIST:
        cx, cy, cz = CAM_MODDER.get_pos(cam)
        CAM_MODDER.set_pos(cam, (cx, cy, cz + obj_h))
      # initialize simulated annealing
      t = 0
      temp = INIT_TEMP
      best_conf, prev_conf = 1.0, 1.0
      score_list, best_pos_list = [], []

    # drop first element at origin
    if cur_obj_handle \
      and idx_obj == 1:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      # move mover to new position
      mover_qpos = _get_object_qpos(cur_obj_name)
      m_cur_x, m_cur_y = m_prev_x, m_prev_y
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(cur_obj_name, mover_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))

      # drop object
      print("Placed %s at %s" % (cur_obj_name, mover_qpos))
      stack_height += obj_h
      cur_obj_handle = None
      obj_orig_qpos = None
      m_prev_x, m_prev_y = m_cur_x, m_cur_y

    # perform simulated annealing with picked object
    if cur_obj_handle \
      and t <= MAX_SA_STEPS:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      # move mover to new position
      mover_qpos = _get_object_qpos(cur_obj_name)
      m_cur_x, m_cur_y = _sa_next_pos(m_prev_x, m_prev_y)
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(cur_obj_name, mover_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))

      # refresh rendering and evaluate position
      MJSIM.step()
      if FLAGS.rendering_mode == 'onscreen':
        mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
      MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
      frame_list = []
      for cam in CAM_LIST:
        frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
        frame = np.flip(frame, 0)
        frame_list.append(frame)
      frames = np.concatenate(frame_list)
      input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
      feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
      scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
      cur_conf = np.average(scores)
      score_list.append((m_cur_x, m_cur_y, cur_conf))

      # accept or reject step based on simulated annealing
      if t % ANNEAL_STEPS: # cool down
        temp *= ANNEAL_SCALE
      a = _sa_acceptance_prob(prev_conf, cur_conf, temp)
      p = np.random.uniform()
      if a > p: # accept the new step
        m_prev_x, m_prev_y = m_cur_x, m_cur_y
        prev_conf = cur_conf
      else: # reject and reset to previous position
        pass

      if cur_conf < best_conf - CONF_INTERVAL: # new best position
        best_conf = cur_conf
        best_pos_list = [(m_cur_x, m_cur_y)]
        print("New best: score=%s, x=%.4f, y=%.4f" % (best_conf, m_cur_x, m_cur_y))
      elif math.fabs(cur_conf - best_conf) < CONF_INTERVAL: # equally good position
        best_pos_list.append((m_cur_x, m_cur_y))
        print("Additional best: score=%s, x=%.4f, y=%.4f" % (best_conf, m_cur_x, m_cur_y))

    # drop current object at best sampled position if stable position had been found
    if cur_obj_handle \
      and t > MAX_SA_STEPS \
      and best_conf < STABILITY_THRES:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      obj_h = obj_init_qpos[2] * 2.0

      # average over best positions
      m_cur_x = np.average([t[0] for t in best_pos_list])
      m_cur_y = np.average([t[1] for t in best_pos_list])
      mover_qpos = _get_object_qpos(cur_obj_name)
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(cur_obj_name, mover_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))

      # DEBUG: show best frame set
      MJSIM.step() # refresh rendering and compute scores
      if FLAGS.rendering_mode == 'onscreen':
        mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
      MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
      frame_list = []
      for cam in CAM_LIST:
        frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
        frame = np.flip(frame, 0)
        frame_list.append(frame)
      frames = np.concatenate(frame_list)
      input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
      feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
      scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
      # plot images
      fig = plt.figure(1, (16., 16.))
      grid = ImageGrid(
          fig, 111,  # similar to subplot(111)
          nrows_ncols=(1, len(frame_list)),
          axes_pad=0.1,  # pad between axes in inch.
          )
      for i, f in enumerate(frame_list):
        title_str = '%s\n%s' % (CAM_LIST[i], str(scores[i]))
        grid[i].set_title(title_str, loc='center')
        grid[i].imshow(f)
      view_img_fn = '%s_best_pos_views_idx=%s.png' % (FLAGS.run_prfx, idx_obj,)
      view_img_path = os.path.join(FLAGS.mjsim_dir, view_img_fn)
      plt.savefig(view_img_path)
      plt.clf()

      # DEBUG: save simulation state
      sim_state_fn = '%s_sim_state_idx=%s.pkl' % (FLAGS.run_prfx, idx_obj,)
      sim_state_path = os.path.join(FLAGS.mjsim_dir, sim_state_fn)
      with open(sim_state_path, 'wb') as f:
        pickle.dump(MJSIM.get_state(), f)

      # DEBUG: compute & visualize score maps
      if len(score_list) > 0:
        X = np.array([t[0] for t in score_list])
        Y = np.array([t[1] for t in score_list])
        Z = np.array([t[2] for t in score_list])

        # DEBUG: show 3D score map
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
        score_img_fn = '%s_score_map_idx=%s.png' % (FLAGS.run_prfx, idx_obj,)
        score_img_path = os.path.join(FLAGS.mjsim_dir, score_img_fn)
        plt.savefig(score_img_path)
        plt.clf()

        # DEBUG: save score list
        score_list_fn = '%s_score_list_idx=%s.pkl' % (FLAGS.run_prfx, idx_obj,)
        score_list_path = os.path.join(FLAGS.mjsim_dir, score_list_fn)
        with open(score_list_path, 'wb') as f:
          pickle.dump(score_list, f)

      # drop object
      print("Placed %s at %s" % (cur_obj_name, mover_qpos))
      stack_height += obj_h
      cur_obj_handle = None
      obj_orig_qpos = None
      m_prev_x, m_prev_y = m_cur_x, m_cur_y

    # put object back and disregard it if confidence threshold can't be met
    if cur_obj_handle \
      and t > MAX_SA_STEPS \
      and best_conf >= STABILITY_THRES:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      obj_h = obj_init_qpos[2] * 2.0
      _set_object_qpos(cur_obj_name, obj_orig_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))
      _deactivate_object(cur_obj_name)

      print("Put back %s to %s as no stable position could be found!" \
        % (cur_obj_name, obj_init_qpos))
      idx_obj -= 1
      cur_obj_handle = None
      obj_orig_qpos = None
      m_prev_x, m_prev_y = 0.0, 0.0

      # reset cameras
      for cam in CAM_LIST:
        cx, cy, cz = CAM_MODDER.get_pos(cam)
        CAM_MODDER.set_pos(cam, (cx, cy, cz - obj_h))

    # advance simulation after applying all changes
    MJSIM.step()
    t += 1

    # render
    if FLAGS.rendering_mode == 'onscreen':
      mujoco_py.functions.mjr_setBuffer(FB_WINDOW, MJSIM.render_contexts[0].con)
      MJVIEWER.render()

def _balancing_mode(stack_base, obj_queue):
  """
  Freezes the current stack while trying to find the correct position of the
  mover object to stabilize / balance the structure.
  """
  print("Performing balancing...")

  # global counters
  idx_obj = 0
  stack_height = 0.0
  stack_collapsed = False
  # local counters (reset after every stack)
  cur_obj_handle = None
  obj_orig_qpos = None
  t = 0
  temp = INIT_TEMP
  best_conf, prev_conf = 1.0, 1.0
  score_list, best_pos_list = [], []
  m_prev_x, m_prev_y = 0.0, 0.0

  # TODO: correctly compute initial stack height!
  for base_obj_name, base_obj_orig_qpos, _ in stack_base:
    base_obj_qpos = _get_object_qpos(base_obj_name)
    stack_height = max( # TODO: hack! only works for cuboid bases!!!
        stack_height,
        base_obj_qpos[2] + _get_object_size(base_obj_name)[2])
    idx_obj += 1
  # adjust cameras
  for cam in CAM_LIST:
    cx, cy, cz = CAM_MODDER.get_pos(cam)
    CAM_MODDER.set_pos(cam, (cx, cy, cz + stack_height))

  while True:

    # terminate after last object has been placed
    if cur_obj_handle is None \
      and t // (MAX_SA_STEPS + STEPS_PLACEMENT) > 0 \
      and len(obj_queue) == 0: # no more objects to stack
      # check for violation
      velocities = np.abs(MJSIM.data.sensordata)
      stack_collapsed = np.any(velocities > VELOCITY_THRES)
      if stack_collapsed:
        print("COLLAPSE!")
      else:
        print("STABLE!")
      # final report
      print("Done stacking objects!")
      if stack_collapsed:
        idx_obj -= 1
      else:
        pass
      print("Final height: %s" % (idx_obj,))
      break

    # check for stability violations while object is settling
    if cur_obj_handle is None \
      and t // (MAX_SA_STEPS + STEPS_PLACEMENT) > 0 \
      and t % STEPS_STACKING_ROUND != 0:
      velocities = np.abs(MJSIM.data.sensordata)
      stack_collapsed = np.any(velocities > VELOCITY_THRES)
      if stack_collapsed:
        obj_queue.clear()

    # pick up next object
    if cur_obj_handle is None \
      and len(obj_queue) > 0 \
      and t % STEPS_STACKING_ROUND == 0:
      # get next object
      cur_obj_handle = obj_queue.pop(0) # (name, init_qpos, score)
      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      obj_orig_qpos = _get_object_qpos(cur_obj_name)
      _activate_object(cur_obj_name)
      idx_obj += 1
      print("Picked up %s" % (cur_obj_name,))
      # compute spawn position
      obj_h = obj_init_qpos[2] * 2.0
      obj_spawn_qpos = np.copy(obj_init_qpos)
      m_cur_x, m_cur_y = _sa_next_pos(m_prev_x, m_prev_y)
      obj_spawn_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      obj_spawn_qpos[2] += (stack_height + OBJ_FLOAT_Z_OFFSET) # adjust spawn height
      # move object into position
      _set_object_qpos(cur_obj_name, obj_spawn_qpos)
      print("Spawned %s at %s" % (cur_obj_name, obj_spawn_qpos))
      # _rotate_object(cur_obj_name, [0.0, 0.0, float(np.random.randint(0, 360))])
      # _rotate_object(cur_obj_name, [0.0, 0.0, 90.0])
      _set_object_vel(cur_obj_name, np.zeros(6))
      # adjust cameras
      for cam in CAM_LIST:
        cx, cy, cz = CAM_MODDER.get_pos(cam)
        CAM_MODDER.set_pos(cam, (cx, cy, cz + obj_h))
      # initialize simulated annealing
      t = 0
      temp = INIT_TEMP
      best_conf, prev_conf = 1.0, 1.0
      score_list, best_pos_list = [], []
      # freeze stack base
      for base_obj_name, base_obj_orig_qpos, _ in stack_base:
        _set_object_qpos(base_obj_name, base_obj_orig_qpos)
        _set_object_vel(base_obj_name, np.zeros(6))

    # drop first element at origin
    if cur_obj_handle \
      and idx_obj == 1:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      # move mover to new position
      mover_qpos = _get_object_qpos(cur_obj_name)
      m_cur_x, m_cur_y = m_prev_x, m_prev_y
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(cur_obj_name, mover_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))

      # drop object
      print("Placed %s at %s" % (cur_obj_name, mover_qpos))
      stack_height += obj_h
      cur_obj_handle = None
      obj_orig_qpos = None
      m_prev_x, m_prev_y = m_cur_x, m_cur_y

    # perform simulated annealing with picked object
    if cur_obj_handle \
      and t <= MAX_SA_STEPS:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      # move mover to new position
      mover_qpos = _get_object_qpos(cur_obj_name)
      m_cur_x, m_cur_y = _sa_next_pos(m_prev_x, m_prev_y)
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(cur_obj_name, mover_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))
      # freeze stack base
      for base_obj_name, base_obj_orig_qpos, _ in stack_base:
        _set_object_qpos(base_obj_name, base_obj_orig_qpos)
        _set_object_vel(base_obj_name, np.zeros(6))

      # refresh rendering and evaluate position
      MJSIM.step()
      if FLAGS.rendering_mode == 'onscreen':
        mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
      MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
      frame_list = []
      for cam in CAM_LIST:
        frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
        frame = np.flip(frame, 0)
        frame_list.append(frame)
      frames = np.concatenate(frame_list)
      input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
      feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
      scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
      cur_conf = np.average(scores)
      score_list.append((m_cur_x, m_cur_y, cur_conf))

      # accept or reject step based on simulated annealing
      if t % ANNEAL_STEPS: # cool down
        temp *= ANNEAL_SCALE
      a = _sa_acceptance_prob(prev_conf, cur_conf, temp)
      p = np.random.uniform()
      if a > p: # accept the new step
        m_prev_x, m_prev_y = m_cur_x, m_cur_y
        prev_conf = cur_conf
      else: # reject and reset to previous position
        pass

      if cur_conf < best_conf - CONF_INTERVAL: # new best position
        best_conf = cur_conf
        best_pos_list = [(m_cur_x, m_cur_y)]
        print("New best: score=%s, x=%.4f, y=%.4f" % (best_conf, m_cur_x, m_cur_y))
      elif math.fabs(cur_conf - best_conf) < CONF_INTERVAL: # equally good position
        best_pos_list.append((m_cur_x, m_cur_y))
        print("Additional best: score=%s, x=%.4f, y=%.4f" % (best_conf, m_cur_x, m_cur_y))

    # drop current object at best sampled position if stable position had been found
    if cur_obj_handle \
      and t > MAX_SA_STEPS \
      and best_conf < STABILITY_THRES:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      obj_h = obj_init_qpos[2] * 2.0

      # average over best positions
      m_cur_x = np.average([t[0] for t in best_pos_list])
      m_cur_y = np.average([t[1] for t in best_pos_list])
      mover_qpos = _get_object_qpos(cur_obj_name)
      mover_qpos[0:2] = np.array([m_cur_x, m_cur_y])
      _set_object_qpos(cur_obj_name, mover_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))

      # DEBUG: show best frame set
      MJSIM.step() # refresh rendering and compute scores
      if FLAGS.rendering_mode == 'onscreen':
        mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
      MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
      frame_list = []
      for cam in CAM_LIST:
        frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
        frame = np.flip(frame, 0)
        frame_list.append(frame)
      frames = np.concatenate(frame_list)
      input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
      feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
      scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
      # plot images
      fig = plt.figure(1, (16., 16.))
      grid = ImageGrid(
          fig, 111,  # similar to subplot(111)
          nrows_ncols=(1, len(frame_list)),
          axes_pad=0.1,  # pad between axes in inch.
          )
      for i, f in enumerate(frame_list):
        title_str = '%s\n%s' % (CAM_LIST[i], str(scores[i]))
        grid[i].set_title(title_str, loc='center')
        grid[i].imshow(f)
      view_img_fn = '%s_best_pos_views_idx=%s.png' % (FLAGS.run_prfx, idx_obj,)
      view_img_path = os.path.join(FLAGS.mjsim_dir, view_img_fn)
      plt.savefig(view_img_path)
      plt.clf()

      # DEBUG: save simulation state
      sim_state_fn = '%s_sim_state_idx=%s.pkl' % (FLAGS.run_prfx, idx_obj,)
      sim_state_path = os.path.join(FLAGS.mjsim_dir, sim_state_fn)
      with open(sim_state_path, 'wb') as f:
        pickle.dump(MJSIM.get_state(), f)

      # DEBUG: compute & visualize score maps
      if len(score_list) > 0:
        X = np.array([t[0] for t in score_list])
        Y = np.array([t[1] for t in score_list])
        Z = np.array([t[2] for t in score_list])

        # DEBUG: show 3D score map
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
        score_img_fn = '%s_score_map_idx=%s.png' % (FLAGS.run_prfx, idx_obj,)
        score_img_path = os.path.join(FLAGS.mjsim_dir, score_img_fn)
        plt.savefig(score_img_path)
        plt.clf()

        # DEBUG: save score list
        score_list_fn = '%s_score_list_idx=%s.pkl' % (FLAGS.run_prfx, idx_obj,)
        score_list_path = os.path.join(FLAGS.mjsim_dir, score_list_fn)
        with open(score_list_path, 'wb') as f:
          pickle.dump(score_list, f)

      # drop object
      print("Placed %s at %s" % (cur_obj_name, mover_qpos))
      stack_height += obj_h
      cur_obj_handle = None
      obj_orig_qpos = None
      m_prev_x, m_prev_y = m_cur_x, m_cur_y

    # put object back and disregard it if confidence threshold can't be met
    if cur_obj_handle \
      and t > MAX_SA_STEPS \
      and best_conf >= STABILITY_THRES:

      cur_obj_name, obj_init_qpos, score = cur_obj_handle
      obj_h = obj_init_qpos[2] * 2.0
      _set_object_qpos(cur_obj_name, obj_orig_qpos)
      _set_object_vel(cur_obj_name, np.zeros(6))
      _deactivate_object(cur_obj_name)

      print("Put back %s to %s as no stable position could be found!" \
        % (cur_obj_name, obj_init_qpos))
      idx_obj -= 1
      cur_obj_handle = None
      obj_orig_qpos = None
      m_prev_x, m_prev_y = 0.0, 0.0

      # reset cameras
      for cam in CAM_LIST:
        cx, cy, cz = CAM_MODDER.get_pos(cam)
        CAM_MODDER.set_pos(cam, (cx, cy, cz - obj_h))

    # advance simulation after applying all changes
    MJSIM.step()
    t += 1

    # render
    if FLAGS.rendering_mode == 'onscreen':
      mujoco_py.functions.mjr_setBuffer(FB_WINDOW, MJSIM.render_contexts[0].con)
      MJVIEWER.render()

def _ranking_mode(obj_queue):
  """
  Places objects from obj_queue from left to right in the world according to \
  their stackability scores.
  Expects obj_queue to be FIFO (pop from front).
  """

  # global counters
  stack_height = 0.0
  # local counters (reset after every stack)
  t = 0
  cur_obj_name = None

  while True:

    if cur_obj_name is None \
      and len(obj_queue) == 0: # no more objects to stack
      break

    if cur_obj_name is None \
      and len(obj_queue) > 0 \
      and t % 2 == 0:
      # get next object from queue
      cur_obj_name, obj_init_qpos, score = obj_queue.pop(0)
      print(cur_obj_name, score)
      # compute spawn position
      spacing = 1.5
      obj_x = -12.0 + len(obj_queue) * (-spacing)
      obj_y = -12.0 + len(obj_queue) * (-spacing)
      obj_init_qpos[0] = obj_x
      obj_init_qpos[1] = obj_y
      # obj_h = obj_init_qpos[2] * 2.0
      # obj_init_qpos[2] += (stack_height + OBJ_FLOAT_Z_OFFSET) # adjust spawn height
      _activate_object(cur_obj_name)
      _set_object_qpos(cur_obj_name, obj_init_qpos)
      # _rotate_object(cur_obj_name, [0.0, 0.0, float(np.random.randint(0, 360))])
      _set_object_vel(cur_obj_name, np.zeros(6))
      # stack_height += obj_h
      cur_obj_name = None

    # advance simulation after applying all changes
    # MJSIM.step()
    t += 1

    # render
    if FLAGS.rendering_mode == 'onscreen':
      mujoco_py.functions.mjr_setBuffer(FB_WINDOW, MJSIM.render_contexts[0].con)
      MJVIEWER.render()

# main function

if __name__ == '__main__':
  # parse input
  FLAGS = ARGPARSER.parse_args()
  print("Simulation with stability estimator in the loop!")
  print("Arguments: ", FLAGS)

  # model and simulation setup
  print("Loading simulation...")
  MJMODEL = mujoco_py.load_model_from_path(FLAGS.mjmodel_path)
  MJSIM = mujoco_py.MjSim(MJMODEL)
  STEPS_PLACEMENT = FLAGS.placement_steps
  VELOCITY_THRES = FLAGS.velocity_thres

  # intialize world
  print("Intializing scene...")
  _init_scene()

  # parse objects from world & initialize
  print("Initializing objects...")
  OBJ_LIST = parse_object_names(MJMODEL)
  if FLAGS.mode == 'stacking':
    _init_objects_stacking(OBJ_LIST)
  else: # leave objects where they are
    pass

  # initialize cameras
  print("Initializing cameras...")
  CAM_LIST = FLAGS.cameras
  STANDBY_CAM = 'cam_16'
  CAM_MODDER = CameraModder(MJSIM)
  _init_cameras(CAM_LIST)

  # quick debug run only
  # if (FLAGS.mode == 'stacking' or FLAGS.mode == 'ranking') and FLAGS.debug:
  #   MJVIEWER = mujoco_py.MjViewer(MJSIM)
  #   _stackability_debug_mode(OBJ_LIST)
  #   _idle_mode()

  # tfmodel & session setup
  print("Initializing stability predictor...")
  SESS, MODEL_ENDPOINTS = _init_stability_predictor(FLAGS.tfmodel, FLAGS.tfckpt_dir)

  # initialize the simulation
  if os.path.isdir(FLAGS.mjsim_dir):
    print("Logging into existing simulation directory: %s" % (FLAGS.mjsim_dir,))
  else: # create new directory
    print("Creating new simulation directory: %s" % (FLAGS.mjsim_dir,))
    os.mkdir(FLAGS.mjsim_dir)

  # set up the viewer
  if FLAGS.rendering_mode == 'onscreen':
    print("Running simulation with on-screen rendering and GUI.")
    MJVIEWER = mujoco_py.MjViewer(MJSIM)
  elif FLAGS.rendering_mode == 'offscreen':
    print("Running simulation with off-screen rendering.")
  else:
    raise Exception("Invalid rendering mode %s" % (FLAGS.rendering_mode,))

  # initialize the simulated annealing
  ANNEAL_STEPS = FLAGS.anneal_steps
  ANNEAL_SCALE = FLAGS.anneal_scale
  MAX_SA_STEPS = FLAGS.max_sa_steps
  MAX_MOVE_XY = FLAGS.max_move_xy
  MAX_POS_XY = FLAGS.max_pos_xy
  STABILITY_THRES = FLAGS.stability_thres

  # set up the stacking queue
  STACKING_QUEUE = [] # (obj_name, init_qpos, stackability_score)
  STACKABILITY = [] # (obj_name, geom_type, init_qpos, area, volume, stackability_score)
  STACK_BASE = [] # (obj_name, init_qpos, stackability_score)
  INIT_QUEUE_PATH = os.path.join(FLAGS.mjsim_dir, 'init_object_queue.pkl')
  RANKING_PATH = os.path.join(FLAGS.mjsim_dir, 'stackability.pkl')

  if FLAGS.mode == 'stacking' or FLAGS.mode == 'ranking':
    if os.path.exists(INIT_QUEUE_PATH): # load stacking queue from file
      with open(INIT_QUEUE_PATH, 'rb') as f:
        STACKING_QUEUE = pickle.load(f)
    else: # run stacking estimation to compute best order of objects
      _stackability_estimation_mode(OBJ_LIST)
      # save computed queue to disk
      with open(INIT_QUEUE_PATH, 'wb') as f:
        pickle.dump(STACKING_QUEUE, f)
      with open(RANKING_PATH, 'wb') as f:
        pickle.dump(STACKABILITY, f)
  elif FLAGS.mode == 'balancing':
    base_obj_names = filter(lambda obj_name: '_base_' in obj_name, OBJ_LIST)
    for base_obj_name in base_obj_names:
      base_init_qpos = _get_object_qpos(base_obj_name)
      STACK_BASE.append((base_obj_name, base_init_qpos, 0.0))
    mover_obj_names = filter(lambda obj_name: '_mover_' in obj_name, OBJ_LIST)
    for mover_obj_name in mover_obj_names:
      mover_init_qpos = _get_object_qpos(mover_obj_name)
      mover_init_qpos[0] = 0.0
      mover_init_qpos[1] = 0.0
      STACKING_QUEUE.append((mover_obj_name, mover_init_qpos, 0.0))
  else:
    raise Exception("Unknown mode %s" % FLAGS.mode)


  # perform task
  if FLAGS.mode == 'stacking':
    _stacking_mode(STACKING_QUEUE)
  elif FLAGS.mode == 'ranking':
    _ranking_mode(STACKING_QUEUE)
  elif FLAGS.mode == 'balancing':
    _balancing_mode(STACK_BASE, STACKING_QUEUE)
  else:
    raise Exception("Unknown mode %s" % FLAGS.mode)

  # run simulation
  if FLAGS.rendering_mode == 'onscreen':
    _idle_mode()
  else:
    # forward scene to let objects settle
    for _ in range(STEPS_STACKING_ROUND):
      MJSIM.step()
    # DEBUG: take final screenshots of the scene
    MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=STANDBY_CAM) # render one dirty image
    frame_list = []
    for cam in CAM_LIST:
      frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
      frame = np.flip(frame, 0)
      frame_list.append(frame)
    input_shape = [len(CAM_LIST), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
    # plot images
    fig = plt.figure(1, (16., 16.))
    grid = ImageGrid(
        fig, 111,  # similar to subplot(111)
        nrows_ncols=(1, len(frame_list)),
        axes_pad=0.1,  # pad between axes in inch.
        )
    for i, f in enumerate(frame_list):
      grid[i].set_title(str(CAM_LIST[i]), loc='center')
      grid[i].imshow(f)
    view_img_fn = '%s_final_views.png' % (FLAGS.run_prfx,)
    view_img_path = os.path.join(FLAGS.mjsim_dir, view_img_fn)
    plt.savefig(view_img_path)
    plt.clf()
    # DEBUG: save simulation state
    sim_state_fn = '%s_sim_state_final.pkl' % (FLAGS.run_prfx,)
    sim_state_path = os.path.join(FLAGS.mjsim_dir, sim_state_fn)
    with open(sim_state_path, 'wb') as f:
      pickle.dump(MJSIM.get_state(), f)
