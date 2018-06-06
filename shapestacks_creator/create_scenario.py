"""
Creates a ShapeStack scenario.
"""

import sys
import os
import argparse
import shutil
import xml.etree.ElementTree as ET
import random
import math

from bs4 import BeautifulSoup

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from simulation_builder.world_builder import MjWorldBuilder
from simulation_builder.asset_builder import MjAssetBuilder
from simulation_builder.mj_schema import MjcfFormat
from simulation_builder.mj_elements import MjVelocimeter, MjLight, MjCamera, \
  MjMaterial, MjGeom
from simulation_builder.mj_templates import MjCuboid, MjCylinder, MjSphere, \
  MjCameraHeadlight


# command line arguments
ARGPARSER = argparse.ArgumentParser(
    description='Create a MuJoCo simulation environment containing an \
    shapestack scenario.')
# model setup and directories
ARGPARSER.add_argument(
    '--mjmodel_name', type=str,
    help="The name of the model to create.")
ARGPARSER.add_argument(
    '--template_path', type=str,
    help="The root directory from which to retrieve a basic MuJoCo model, \
    asset catalog and world setup.")
ARGPARSER.add_argument(
    '--export_path', type=str,
    help="The root directory to which the created world and all assets will \
    be exported.")
ARGPARSER.add_argument(
    '--overwrite_assets', action='store_true',
    help="Overwrite assets already present in export directory.")
# stack geometry
ARGPARSER.add_argument(
    '--shapes', type=str, nargs='+', default=['cuboid', 'cylinder', 'sphere'],
    help="The available shapes for stacking.")
ARGPARSER.add_argument(
    '--obj_dim_min', type=float, default=0.5,
    help="Minimum size per object dimension.")
ARGPARSER.add_argument(
    '--obj_dim_max', type=float, default=1.0,
    help="Maximum size per object dimension.")
ARGPARSER.add_argument(
    '--height', type=int, default=6,
    help="The number of blocks to stack.")
ARGPARSER.add_argument(
    '--vcom', type=int, nargs='+', default=[0],
    help="The index numbers of the objects which break the stability of the \
    tower by violating the COM principle (not supporting the COM of the above \
    stack). Starting from 1 (bottom object) to height - 1 (second to top \
    object). Index 0 means stable stack.")
ARGPARSER.add_argument(
    '--com_scale_min', type=float, default=0.00,
    help="Inner radius for COM surface sampling.")
ARGPARSER.add_argument(
    '--com_scale_max', type=float, default=0.50,
    help="Outer radius for COM surface sampling")
ARGPARSER.add_argument(
    '--vcom_scale', type=float, default=0.05,
    help="The absolute offset added to an allowed COM offset during \
    violation.")
ARGPARSER.add_argument(
    '--vpsf', type=int, nargs='+', default=[0],
    help="The index numbers of the objects which break the stability of the \
    tower by violating the flat surface principle (round surface underneath \
    top of stack). Starting from 0 (bottom object) to height - 1 (second to top \
    object). Index 0 means stable stack.")
# rendering setup, deprecated, mostly controlled during rendering
ARGPARSER.add_argument(
    '--simple_world', action='store_true',
    help="Creates a simplistic world with a grey ground plane only.")
ARGPARSER.add_argument( # deprecated, changed during rendering
    '--floortex', type=int, default=0,
    help="The index number of the floor texture to use (zero based).")
ARGPARSER.add_argument( # deprecated, changed during rendering
    '--walltex', type=int, default=0,
    help="The index number of the wall texture to use (zero based).")
ARGPARSER.add_argument(
    '--lightids', type=int, nargs='+', default=[],
    help="The index numbers of the scene lights to be used as shadow casting\
    main lights (zero based).")
ARGPARSER.add_argument(
    '--camids', type=int, nargs='+', default=[],
    help="The index numbers of the cameras to be used (zero based). If not\
    specified, all cameras will be used.")
ARGPARSER.add_argument( # deprecated
    '--camlights_on', action='store_true',
    help="Adds headlights to all cameras.")
ARGPARSER.add_argument(
    '--numcolors', type=int, default=6,
    help="The number of colors to use for the stacked blocks (max. 6).")
ARGPARSER.add_argument(
    '--unique_colors', action='store_true',
    help="Use unique shape colors, if possible. Limited by numcolors")

# model parts under <mjmodel_name>/
# env_<mjmodel_name>.xml: top-level model, compiler and simulation options, include hooks
# assets_<mjmodel_name>.xml: asset catalog
# world_<mjmodel_name>.xml: basic world scaffolding
# textures/ : texture directory
# meshes/ : mesh directory


# CONSTANTS

# static world geometry
PLANE_L = 10
PLANE_H = 0.5
PLANE_NAMES = [
    'floor',
    'wall_1',
    'wall_2',
]
PLANE_POSITIONS = [
    (0, 0, 0),                # ground plate
    (0, PLANE_L, PLANE_L),    # wall 1
    (PLANE_L, 0, PLANE_L),    # wall 2
]
PLANE_SIZES = [
    (PLANE_L, PLANE_L, PLANE_H),    # ground plate
    (PLANE_L, PLANE_L, PLANE_H),    # wall 1
    (PLANE_L, PLANE_L, PLANE_H),    # wall 2
]
PLANE_EULERS = [
    (0, 0, 0),      # ground plate
    (90, 180, 0),   # wall 1
    (270, 0, 90),   # wall 2
]

# objects
OBJ_COLORS_RGBA = [
    (1, 0, 0, 1),  # red
    (0, 1, 0, 1),  # green
    (0, 0, 1, 1),  # blue
    (1, 1, 0, 1),  # yellow
    (0, 1, 1, 1),  # cyan
    (1, 0, 1, 1),  # magenta
]

# stack
STACK_ORIGIN = (0.0, 0.0)
ORIGIN_OFFSET_MAX = 2.0

# light setup
LIGHT_POSITIONS = [
    (0, 0, 20),     # light_0 (top light)
    (-9, -9, 20),   # light_1 (corner 1)
    (-9, 9, 20),    # light_2 (corner 2)
    (9, 9, 20),     # light_3 (corner 3)
    (9, -9, 20),    # light_4 (corner 4)
]
LIGHT_DIRECTIONS = [
    (0, 0, -20),
    (9, 9, -20),
    (9, -9, -20),
    (-9, -9, -20),
    (-9, 9, -20),
]

# camera setup
CAMERA_POSITIONS = [
    # corner 1
    (-7, -7, 5),   # cam_1: center
    (-9, -2, 5),   # cam_2: left
    (-2, -9, 5),   # cam_3: right
    # corner 2
    (-7, 7, 5),   # cam_4 center
    (-9, 2, 5),   # cam_5: right
    (-2, 9, 5),   # cam_6: left
    # corner 3
    (7, 7, 5),    # cam_7: center
    (9, 2, 5),    # cam_8: left
    (2, 9, 5),    # cam_9: right
    # corner 4
    (7, -7, 5),   # cam_10: center
    (9, -2, 5),   # cam_11: right
    (2, -9, 5),   # cam_12: left
    # top
    (-5, -5, 9),  # cam_13: corner 1
    (-5, 5, 9),   # cam_14: corner 2
    (5, 5, 9),    # cam_15: corner 3
    (5, -5, 9),   # cam_16: corner 4
]
CAMERA_EULERS = [
    # corner 1
    (75, 0, -45),
    (75, 0, -75),
    (75, 0, -15),
    # corner 2
    (75, 0, 225),
    (75, 0, 255),
    (75, 0, 195),
    # corner 3
    (75, 0, 135),
    (75, 0, 105),
    (75, 0, 165),
    # corner 4
    (75, 0, 45),
    (75, 0, 75),
    (75, 0, 15),
    # top
    (45, 0, -45),
    (45, 0, 225),
    (45, 0, 135),
    (45, 0, 45),
]
# CAMLIGHT_DIRECTIONS = [
#     (8, 8, -3),
#     (8, -8, -3),
#     (-8, -8, -3),
#     (-8, 8, -3),
#     # (10, 2, -3),
#     # (2, 10, -3),
#     # (6, 6, -7),
#     # (10, 10, 0),
# ]


# scenario creation subroutines

def create_materials(ab: MjAssetBuilder):
  """
  Creates the materials based on the available textures.
  """
  texture_names = ab.get_texture_names()
  texture_categories = ['floor', 'wall']
  for tex_cat in texture_categories:
    for tex_name in \
      filter(lambda tn: tn.startswith('tex_' + tex_cat), texture_names):
      mat = MjMaterial()
      mat.name = 'mat_' + tex_name.lstrip('tex_')
      mat.texture = tex_name
      # if tex_cat == 'floor':
      #     mat.texuniform = "true"
      # mat.texuniform = "true"
      ab.add_asset(mat)

def create_world(wb: MjWorldBuilder, ab: MjAssetBuilder, mat_index):
  """
  Assembles the static world body.
  """
  for p_name, p_pos, p_size, p_euler in \
      zip(PLANE_NAMES, PLANE_POSITIONS, PLANE_SIZES, PLANE_EULERS):
    # create plane
    plane = MjGeom()
    plane.type = "plane"
    plane.name = p_name
    plane.pos = MjcfFormat.tuple(p_pos)
    plane.size = MjcfFormat.tuple(p_size)
    plane.euler = MjcfFormat.tuple(p_euler)
    # select material and assign to plane
    p_cat = p_name.split('_')[0] # first name element indicates category
    mat_idx = mat_index[p_cat]
    mat_prfx = 'mat_' + p_cat
    mat_filter = [n for n in ab.get_material_names() if n.startswith(mat_prfx)]
    mat_filter = sorted(mat_filter)
    mat_name = mat_filter[mat_idx]
    plane.material = mat_name
    # insert plane
    wb.insert_static(plane)

def create_simple_world(wb: MjWorldBuilder):
  """
  Inserts the ground plate for a simplistic world.
  """
  # create plane
  plane = MjGeom()
  plane.type = "plane"
  plane.name = PLANE_NAMES[0]
  plane.pos = MjcfFormat.tuple(PLANE_POSITIONS[0])
  plane.size = MjcfFormat.tuple(PLANE_SIZES[0])
  plane.euler = MjcfFormat.tuple(PLANE_EULERS[0])
  plane.rgba = MjcfFormat.tuple((.4, .4, .4, 1.0))
  # insert plane
  wb.insert_static(plane)

def _create_cuboid():
  obj = MjCuboid()
  len_x = random.uniform(FLAGS.obj_dim_min, FLAGS.obj_dim_max)
  len_y = random.uniform(FLAGS.obj_dim_min, FLAGS.obj_dim_max)
  len_z = random.uniform(FLAGS.obj_dim_min, FLAGS.obj_dim_max)
  obj.set_size(len_x, len_y, len_z)
  return obj

def _create_cylinder():
  obj = MjCylinder()
  len_r = random.uniform(FLAGS.obj_dim_min, FLAGS.obj_dim_max) / 2
  len_h = random.uniform(FLAGS.obj_dim_min, FLAGS.obj_dim_max)
  obj.set_size(len_r, len_h)
  return obj

def _create_sphere():
  obj = MjSphere()
  len_r = random.uniform(FLAGS.obj_dim_min, FLAGS.obj_dim_max) / 2
  obj.set_size(len_r)
  return obj

def create_stack(
    wb: MjWorldBuilder,
    shapes, height: int,
    num_colors: int, unique_colors: bool,
    com_violations, surface_violations,
    com_scale_min: float, com_scale_max: float,
    vcom_scale: float):
  """
  Creates the shape stack with the defined instabilities (based on COM and
  surfaces).
  """

  # color setup
  obj_colors = OBJ_COLORS_RGBA
  random.shuffle(obj_colors)
  obj_colors = obj_colors[:num_colors]

  # shape setup
  all_shapes = []
  stable_shapes = []
  unstable_shapes = []
  if 'cuboid' in shapes:
    all_shapes.append('cuboid')
    stable_shapes.append('cuboid')
  if 'cylinder' in shapes:
    all_shapes.append('cylinder_standing')
    all_shapes.append('cylinder_lying')
    stable_shapes.append('cylinder_standing')
    unstable_shapes.append('cylinder_lying')
  if 'sphere' in shapes:
    all_shapes.append('sphere')
    unstable_shapes.append('sphere')

  # center of mass
  com_x, com_y = STACK_ORIGIN
  cur_mass = 0.0 # mass = volume since all objects have equal density
  # shapes
  shape_list = []
  # stack
  last_x, last_y = STACK_ORIGIN # coordinates of previous object
  stack_height = 0.0

  # create stack from top to bottom
  for i in range(height):

    # rotation angle of new object
    angle = random.randint(0, 360)

    # check for surface violation during object creation
    if i == 0:
      # all shapes allowed on top
      rnd_idx = random.randint(0, len(all_shapes)-1)
      selector = all_shapes[rnd_idx]
      if selector == 'cuboid':
        obj = _create_cuboid()
        obj.euler = MjcfFormat.tuple((0, 0, angle))
        x, y, z = obj.get_size()
        V = x * y * z
        surface = ('rectangle', x, y)
      elif selector == 'cylinder_lying':
        obj = _create_cylinder()
        obj.euler = MjcfFormat.tuple((90, 0, angle))
        r, h = obj.get_size()
        x = 2 * r
        y = h
        z = 2 * r
        V = math.pi * (r ** 2) * h
        surface = ('line', 0.0, h)
      elif selector == 'cylinder_standing':
        obj = _create_cylinder()
        angle = 0 # rotation symmetric
        r, h = obj.get_size()
        x = 2 * r
        y = 2 * r
        z = h
        V = math.pi * (r ** 2) * h
        surface = ('circle', r, r)
      elif selector == 'sphere':
        obj = _create_sphere()
        angle = 0 # rotation symmetric
        r = obj.get_size()[0]
        x = r * 2
        y = r * 2
        z = r * 2
        V = (4 / 3) * math.pi * (r ** 3)
        surface = ('point', 0.0, 0.0)
      else:
        raise Exception('Invalid shape %s' % selector)
    elif height - i in surface_violations:
      # randomly create sphere or lying cylinder
      rnd_idx = random.randint(0, len(unstable_shapes)-1)
      selector = unstable_shapes[rnd_idx]
      if selector == 'sphere':
        obj = _create_sphere()
        angle = 0 # rotation symmetric
        r = obj.get_size()[0]
        x = r * 2
        y = r * 2
        z = r * 2
        V = (4 / 3) * math.pi * (r ** 3)
        surface = ('point', 0.0, 0.0)
      elif selector == 'cylinder_lying':
        obj = _create_cylinder()
        obj.euler = MjcfFormat.tuple((90, 0, angle))
        r, h = obj.get_size()
        x = 2 * r
        y = h
        z = 2 * r
        V = math.pi * (r ** 2) * h
        surface = ('line', 0.0, h)
      else:
        raise Exception('Invalid shape %s' % selector)
    else: # normal object without surface violation
      # randomly create cuboid or standing cylinder
      rnd_idx = random.randint(0, len(stable_shapes)-1)
      selector = stable_shapes[rnd_idx]
      if selector == 'cuboid':
        obj = _create_cuboid()
        obj.euler = MjcfFormat.tuple((0, 0, angle))
        x, y, z = obj.get_size()
        V = x * y * z
        surface = ('rectangle', x, y)
      elif selector == 'cylinder_standing':
        obj = _create_cylinder()
        angle = 0 # rotation symmetric
        r, h = obj.get_size()
        x = 2 * r
        y = 2 * r
        z = h
        V = math.pi * (r ** 2) * h
        surface = ('circle', r, r)
      else:
        raise Exception('Invalid shape %s' % selector)

    # assign color
    if FLAGS.unique_colors:
      obj.rgba = MjcfFormat.tuple(obj_colors[i % len(obj_colors)])
    else:
      obj.rgba = MjcfFormat.tuple(obj_colors[random.randint(0, len(obj_colors)-1)])

    # check for COM violation during offset computation
    if i == 0:
      # place first object at origin
      off_x, off_y = 0.0, 0.0
    elif height - i in com_violations:
      # put the object beyond the max offset for stable COM stacking in either dimension
      # sample beyond the bounds of the current surface
      surface_type = surface[0]
      x, y = surface[1], surface[2]
      mode = random.randint(0, 2)
      if  mode < 1: # violate off_x
        off_x = (-1) ** random.randint(0, 1) * (x / 2 + vcom_scale)
        off_y = random.uniform(-(y / 2), y / 2)
      elif mode < 2: # violate off_y
        off_x = random.uniform(-(x / 2), x / 2)
        off_y = (-1) ** random.randint(0, 1) * (y / 2 + vcom_scale)
      else: # violate off_x and off_y
        off_x = (-1) ** random.randint(0, 1) * (x / 2 + vcom_scale)
        off_y = (-1) ** random.randint(0, 1) * (y / 2 + vcom_scale)
    else: # put the block in a stable position (within stable offset from COM)
      # sample within the bounds of the current surface
      surface_type = surface[0]
      x, y = surface[1], surface[2]
      min_x, max_x = (x / 2) * com_scale_min, (x / 2) * com_scale_max
      min_y, max_y = (y / 2) * com_scale_min, (y / 2) * com_scale_max
      off_x = (-1) ** random.randint(0, 1) * random.uniform(min_x, max_x)
      off_y = (-1) ** random.randint(0, 1) * random.uniform(min_y, max_y)
      if surface_type == 'rectangle':
        pass
      elif surface_type == 'circle':
        r = x
        max_r = r * com_scale_max
        while True: # rejection sampling within circle
          if (off_x ** 2) + (off_y ** 2) < (max_r ** 2):
            break
          else:
            off_x = (-1) ** random.randint(0, 1) * random.uniform(min_x, max_x)
            off_y = (-1) ** random.randint(0, 1) * random.uniform(min_y, max_y)
            continue
      elif surface_type == 'line':
        pass
      elif surface_type == 'point':
        pass
      else:
        raise Exception("Unknown surface type %s" % surface_type)

    # rotate offsets according to current object rotation
    tmp_x, tmp_y = off_x, off_y
    rad = (float(angle) / 360) * (2 * math.pi)
    off_x = tmp_x * math.cos(rad) - tmp_y * math.sin(rad)
    off_y = tmp_x * math.sin(rad) + tmp_y * math.cos(rad)

    # apply offsets
    pos_x = com_x + off_x
    pos_y = com_y + off_y
    off_z = stack_height - z / 2
    pos = (pos_x, pos_y, off_z)
    obj.name = "shape_%s" % (height - i)
    obj.pos = MjcfFormat.tuple(pos)
    stack_height -= z
    shape_list.append(obj)

    # memorize position of current object for further reference
    last_x, last_y = pos_x, pos_y

    # update center of mass
    com_x = (V * pos_x + cur_mass * com_x) / (V + cur_mass)
    com_y = (V * pos_y + cur_mass * com_y) / (V + cur_mass)
    cur_mass += V

  # final adjustment of tower
  orig_off_x = random.uniform(-ORIGIN_OFFSET_MAX, ORIGIN_OFFSET_MAX)
  orig_off_y = random.uniform(-ORIGIN_OFFSET_MAX, ORIGIN_OFFSET_MAX)
  for obj in shape_list:
    # adjust height
    xpos, ypos, zpos = MjcfFormat.strtuple2float(obj.pos)
    obj.pos = MjcfFormat.tuple((
        xpos + orig_off_x, ypos + orig_off_y, zpos + math.fabs(stack_height)))
    # insert object
    wb.insert_dynamic(obj)

def create_instrumentation(wb: MjWorldBuilder, height: int):
  """
  Adds the instrumentation for collapse measurement to the world space.
  """
  velo = MjVelocimeter()
  velo.name = "velo_collapse"
  wb.attach_sensor(velo, "shape_%s" % height)

def create_light(wb: MjWorldBuilder, light_id: int, castshadow: bool):
  """
  Inserts a world light (global scene light).
  """
  light = MjLight()
  light.name = "light_" + str(light_id)
  light.directional = "true"
  light.pos = MjcfFormat.tuple(LIGHT_POSITIONS[light_id])
  light.dir = MjcfFormat.tuple(LIGHT_DIRECTIONS[light_id])
  if castshadow:
    light.castshadow = "true"
  else:
    light.castshadow = "false"
  wb.insert_static_light(light)

def create_camera(wb: MjWorldBuilder, cam_id: int, with_headlight: bool = False):
  """
  Inserts the static cameras.
  """
  # camera
  cam = MjCamera()
  cam.name = "cam_%s" % (cam_id + 1)
  cam.pos = MjcfFormat.tuple(CAMERA_POSITIONS[cam_id])
  cam.euler = MjcfFormat.tuple(CAMERA_EULERS[cam_id])
  wb.insert_static_camera(cam)
  # camera headlight
  # if with_headlight:
  #   cam_light = MjCameraHeadlight()
  #   cam_light.name = "cam_light_%s" % (cam_id + 1)
  #   cam_light.pos = MjcfFormat.tuple(CAMERA_POSITIONS[cam_id])
  #   cam_light.dir = MjcfFormat.tuple(CAMLIGHT_DIRECTIONS[cam_id])
  #   cam_light.ambient = MjcfFormat.tuple((0.0, 0.0, 0.0))
  #   cam_light.diffuse = MjcfFormat.tuple((0.7, 0.7, 0.7))
  #   cam_light.specular = MjcfFormat.tuple((0.3, 0.3, 0.3))
  #   wb.insert_static_light(cam_light)


if __name__ == '__main__':
  # parse input
  FLAGS = ARGPARSER.parse_args()
  print("Creating a ShapeStack scenario!")
  print("Arguments: ", FLAGS)

  # builder setup
  ab = MjAssetBuilder()
  ab.load_assets(os.path.join(FLAGS.template_path, 'assets.xml'))
  wb = MjWorldBuilder()
  wb.load_world(os.path.join(FLAGS.template_path, 'world.xml'))

  # load & convert the raw assets
  if not os.path.exists(FLAGS.export_path) or FLAGS.overwrite_assets:
    shutil.copytree(FLAGS.template_path, FLAGS.export_path)
    catalog_texdir = os.path.join(FLAGS.template_path, 'textures')
    ab.convert_textures(catalog_texdir)
    create_materials(ab)
  else:
    ab.load_assets(os.path.join(FLAGS.export_path, 'assets.xml'))

  # insert static world
  if FLAGS.simple_world:
    create_simple_world(wb)
  else:
    mat_index = {}
    mat_index.update({'floor' : FLAGS.floortex})
    mat_index.update({'wall' : FLAGS.walltex})
    create_world(wb, ab, mat_index)

  # stack objects
  create_stack(
      wb,
      FLAGS.shapes, FLAGS.height,
      FLAGS.numcolors, FLAGS.unique_colors,
      FLAGS.vcom, FLAGS.vpsf,
      FLAGS.com_scale_min, FLAGS.com_scale_max,
      FLAGS.vcom_scale)
  create_instrumentation(wb, FLAGS.height)

  # add lights
  for light_id in range(len(LIGHT_POSITIONS)):
    if light_id in FLAGS.lightids: # set as main light to cast shadow
      create_light(wb, light_id, castshadow=True)
    else:
      create_light(wb, light_id, castshadow=False)

  # add cameras
  if len(FLAGS.camids) == 0:
    cam_id_list = list(range(len(CAMERA_POSITIONS)))
  else:
    cam_id_list = FLAGS.camids
  for cam_id in cam_id_list:
    create_camera(wb, cam_id, FLAGS.camlights_on)

  # export
  env = ET.parse(os.path.join(FLAGS.template_path, 'env.xml')).getroot()
  env.attrib["model"] = FLAGS.mjmodel_name
  world_include = env.find(".//include[@file='world.xml']")
  world_include.attrib["file"] = 'world_%s.xml' % FLAGS.mjmodel_name
  env_xml_file = os.path.join(FLAGS.export_path, 'env_%s.xml' % FLAGS.mjmodel_name)
  with open(env_xml_file, 'w') as f:
    xml_str = BeautifulSoup(ET.tostring(env), 'xml').prettify()
    f.write(xml_str)
  ab.export_assets(os.path.join(FLAGS.export_path, 'assets.xml'))
  wb.export_world(os.path.join(FLAGS.export_path, 'world_%s.xml' % FLAGS.mjmodel_name))
