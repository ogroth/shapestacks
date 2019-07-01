"""
Contains utility functions to parse ShapeStacks files.
"""

from collections import namedtuple
import os
import xml.etree.ElementTree as et
import re


# ---------- ad-hoc data structures ----------

ShapeStacksObject = namedtuple(
    'ShapeStacksObject',
    [
        'scenario',  # name of the ShapeStacks scenario this shape is extracted from
        'name',  # name of the shape in the scenario
        'com',  # cartesian coordinates of the CoM in the worldspace
        'euler',  # x-y-z (degrees) euler orientation of the shape
        'level',  # level of the shape within the stack (1 = lowest)
        'shape',  # type of the shape, \in {box, cylinder, sphere}
        'size',  # spatial dimensions, semantics varies according to type
        'rgba'  # RGBA color of the shape (as float32)
    ]
)


# ---------- helper functions ----------

def _strtuple2float(strtup):
  """Converts a tuple string to a tuple of floats."""
  return tuple([float(t) for t in strtup.split(' ')])


# ---------- public APIs ----------

def extract_objects(worldfile_path: str):
  """
  Extracts information about all objects from an XML defining a ShapeStacks world.
  Example file: data/shapestacks_example/mjcf/world_ccs-easy-h=5-vcom=2-vpsf=0-v=1.xml

  Returns:
    [ShapeStacksObject]
  """

  # parse world
  scenario, _ = os.path.splitext(os.path.basename(worldfile_path))
  world = et.parse(worldfile_path).getroot()
  worldbody = world.find('worldbody')
  body_nodes = worldbody.findall('body')

  # parse each body into a ShapeStacksObject
  shapestacks_objects = []
  for body_node in body_nodes:
    geom_node = body_node.find('geom')
    name = geom_node.attrib['name']
    com = _strtuple2float(body_node.attrib['pos'])
    euler = _strtuple2float(body_node.attrib['euler'])
    level = int(re.findall(r'\d+', name)[0])
    shape = geom_node.attrib['type']
    size = _strtuple2float(geom_node.attrib['size'])
    rgba = _strtuple2float(geom_node.attrib['rgba'])
    shapestacks_objects.append(
        ShapeStacksObject(
            scenario=scenario, name=name,
            com=com, euler=euler, level=level,
            shape=shape, size=size,
            rgba=rgba
        )
    )

  return shapestacks_objects
