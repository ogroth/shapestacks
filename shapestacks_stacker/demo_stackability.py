"""
Run demo script for stackability.

On Ubuntu 16.04 execute with for offscreen rendering:
LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so
"""

import sys
import os
import argparse
import re

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm

import mujoco_py
from mujoco_py.modder import CameraModder, TextureModder
from mujoco_py.generated.const import FB_OFFSCREEN, FB_WINDOW

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from tf_models.inception.inception_v4 import inception_v4
from utilities.rotation_utils import quaternion_from_euler, euler_from_quaternion
from utilities.mujoco_utils import mjsim_mat_id2name, mjhlp_geom_type_id2name

# command line arguments
ARGPARSER = argparse.ArgumentParser(
    description='Run demo script for stackability.')
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
# stacking mode
ARGPARSER.add_argument(
    '--mode', type=str, default='rotation_demo',
    help="Run mode of stacker. Available: rotation_demo.")
# camera parameters
ARGPARSER.add_argument(
    '--cameras', type=str, nargs='+',
    default=['cam_1', 'cam_4', 'cam_7', 'cam_10', 'cam_13', 'cam_15'],
    help="The cameras to observe the scene with (cam_1 through cam_16).")
# rendering parameters
ARGPARSER.add_argument(
    '--rendering_mode', type=str, default="onscreen",
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

# object constants
OBJ_COLORS_RGBA = [
    [1, 0, 0, 1],  # red
    [0, 1, 0, 1],  # green
    [0, 0, 1, 1],  # blue
    [1, 1, 0, 1],  # yellow
    [0, 1, 1, 1],  # cyan
    [1, 0, 1, 1],  # magenta
]


# simulation functions

def _parse_object_names(mjmodel):
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

def _get_object_color(obj_name: str):
  """
  Returns the object color as normalized RGBA array.
  """
  mj_obj_id = MJMODEL.geom_name2id(obj_name)
  obj_color = MJMODEL.geom_rgba[mj_obj_id]
  return obj_color

def _set_object_color(obj_name: str, obj_color):
  """
  Sets the object color to the specified normalized RGBA array.
  """
  mj_obj_id = MJMODEL.geom_name2id(obj_name)
  obj_color = MJMODEL.geom_rgba[mj_obj_id]
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


# stability predictor

def _evaluate_stability(active_cameras, standby_camera):
  """
  Evaluates the stability of the scene from the given camera angles.
  """
  if FLAGS.rendering_mode == 'onscreen':
    mujoco_py.functions.mjr_setBuffer(FB_OFFSCREEN, MJSIM.render_contexts[0].con)
  MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=standby_camera)  # render one dirty image
  frame_list = []
  for cam in active_cameras:
    frame = MJSIM.render(RENDER_HEIGHT, RENDER_WIDTH, camera_name=cam)
    frame = np.flip(frame, 0)
    frame_list.append(frame)
  frames = np.concatenate(frame_list)
  input_shape = [len(active_cameras), RENDER_HEIGHT, RENDER_WIDTH, RENDER_CHANNELS]
  feed_dict = {MODEL_ENDPOINTS['inputs'] : np.reshape(frames, input_shape)}
  scores = SESS.run([MODEL_ENDPOINTS['scores'],], feed_dict=feed_dict)[0]
  stability = np.average(scores)
  return stability


# demo tasks

def _demo_rotation(obj_name):
  """
  Places the object for `obj_name` in scene center and rotates it.
  """
  # object initialization
  _OBJ_OFFSET = np.sqrt(0.75) - 0.5
  _activate_object(obj_name)
  obj_shape = _get_object_shape(obj_name)
  obj_size = _get_object_size(obj_name)
  obj_init_pos = [0, 0, obj_size[-1] + _OBJ_OFFSET]
  _set_object_pos(obj_name, obj_init_pos)

  # stability visualization
  colormap = cm.get_cmap('RdYlGn')
  # colormap = cm.get_cmap('coolwarm')
  texture_modder = TextureModder(MJSIM)

  # rotation
  _ANGLE_RESOLUTION = 1
  _SINGLE_AXES = [0, 1, 2]
  _DOUBLE_AXES = [(0, 1), (0, 2), (1, 2)]

  while True:
    for axis in _SINGLE_AXES:  # rotate around single axes
      for angle in range(0, 360 * _ANGLE_RESOLUTION, 1):
        # set angle
        euler = [0.0, 0.0, 0.0]
        euler[axis] = float(angle) / _ANGLE_RESOLUTION
        _set_object_euler(obj_name, euler)
        _set_object_pos(obj_name, obj_init_pos)
        # compute stability
        stability = _evaluate_stability(CAM_LIST, STANDBY_CAM)
        # set color
        # rad = float(angle) / _ANGLE_RESOLUTION / 360.0 * 2 * np.pi
        # color_idx = 0.5 * (np.cos(4 * rad) + 1.0)
        color_idx = 1.0 - stability
        obj_color = np.array(colormap(color_idx))
        texture_modder.set_rgb(obj_name, obj_color[0 : 3] * 256.0)
        _step_and_render()
    for axis in _DOUBLE_AXES:  # rotate around double axes
      for angle in range(0, 360 * _ANGLE_RESOLUTION, 1):
        # set angle
        euler = [0.0, 0.0, 0.0]
        euler[axis[0]] = float(angle) / _ANGLE_RESOLUTION
        euler[axis[1]] = float(angle) / _ANGLE_RESOLUTION
        _set_object_euler(obj_name, euler)
        _set_object_pos(obj_name, obj_init_pos)
        # compute stability
        stability = _evaluate_stability(CAM_LIST, STANDBY_CAM)
        # set color
        # rad = float(angle) / _ANGLE_RESOLUTION / 360.0 * 2 * np.pi
        # color_idx = 0.5 * (np.cos(4 * rad) + 1.0)
        color_idx = 1.0 - stability
        obj_color = np.array(colormap(color_idx))
        texture_modder.set_rgb(obj_name, obj_color[0 : 3] * 256.0)
        _step_and_render()


# simulation modes

def _step_and_render():
  """
  Forwards the simulation by one timestep and renders this step.
  """
  MJSIM.step()
  if FLAGS.rendering_mode == 'onscreen':
    mujoco_py.functions.mjr_setBuffer(FB_WINDOW, MJSIM.render_contexts[0].con)
    MJVIEWER.render()

def _idle_mode():
  """
  Just lets the simulation idle until it is terminated.
  """
  while True:
    _step_and_render()


# main function

if __name__ == '__main__':
  # parse input
  FLAGS = ARGPARSER.parse_args()
  print("Running stackability demo.")
  print("Arguments: ", FLAGS)

  # model and simulation setup
  print("Loading simulation...")
  MJMODEL = mujoco_py.load_model_from_path(FLAGS.mjmodel_path)
  MJSIM = mujoco_py.MjSim(MJMODEL)

  # parse objects from world & initialize
  print("Initializing objects...")
  OBJ_LIST = _parse_object_names(MJMODEL)

  # initialize cameras
  print("Initializing cameras...")
  CAM_LIST = FLAGS.cameras
  STANDBY_CAM = 'cam_16'
  CAM_MODDER = CameraModder(MJSIM)
  _init_cameras(CAM_LIST)

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

  # execute the demo task
  _demo_rotation(OBJ_LIST[0])

  # simulation keep simulation running
  if FLAGS.rendering_mode == 'onscreen':
    _idle_mode()
  else:
    pass
