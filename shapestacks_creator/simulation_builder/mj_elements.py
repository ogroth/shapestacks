"""
Defines the Python mappings of MuJoCo model elements. TODO: Auto-generate
Mj*-types based on an XSD parse of the MJCF schema definition! Possible with
dynamic type creation: https://docs.python.org/3.5/library/types.html
"""

import xml.etree.ElementTree as ET
from simulation_builder.mj_schema import MjcfElement


# abstract base mapping

class MjBaseElement(MjcfElement):
  """
  Abstact base element for mapping of Python objects to MJCF schema nodes.
  Provides all basic interfaces to nest objects, read from and parse to XML.
  """

  def __init__(self, nodename, attribute_names, children):
    attributes = dict([(att, "") for att in attribute_names])
    super(MjBaseElement, self).__init__(nodename, attributes, children)

  def add_child_elem(self, child):
    """
    Add a MjcfElement child to the current MjcfElement node.
    """
    if not isinstance(child, MjcfElement):
      raise TypeError("Children of a MjcfElement must be a MjcfElement themselves!")
    else:
      self._children.append(child)

  def get_children(self):
    """
    Returns all child elements of the MjcfElement.
    """
    return self._children

  def to_etree_elem(self):
    """
    Returns an xml.etree.ElementTree.Element representing the MjcfElement.
    """
    elem = ET.Element(self._nodename)
    for att, val in self._attributes.items():
      if not val == "":
        elem.set(att, val)
    for child in self._children:
      elem.append(child.to_etree_elem())
    return elem

  @staticmethod
  def from_etree_elem(etree_elem):
    """
    Static factory method for creation of subclasses of MjcfElement.
    """
    # mapping of nodenames to Python classes
    mjcf_element_types = dict(
        [(t.__name__.lstrip("Mj").lower(), t) for t in MjBaseElement.__subclasses__()])
    # create root node from nodename
    nodename = etree_elem.tag
    if nodename in mjcf_element_types:
      root = mjcf_element_types[nodename]()
    else:
      raise TypeError("Nodename %s has no Python class mapping!" % nodename)
    # set attributes of root
    for att, val in etree_elem.attrib.items():
      root.__setattr__(att, val)
    # append all children recursively
    for etree_child in etree_elem:
      child = MjBaseElement.from_etree_elem(etree_child)
      root.add_child_elem(child)
    return root


# model

class MjMujoco(MjBaseElement):
  """
  Python equivalent of MJCF mujoco. The unique top-level element of every
  model.
  """

  def __init__(self):
    nodename = "mujoco"
    attribute_names = ["model"]
    children = []
    super(MjVisual, self).__init__(nodename, attribute_names, children)

class MjInclude(MjBaseElement):
  """
  Python equivalent of MJCF include. The meta-element to include sub-files
  of the model description.
  """

  def __init__(self):
    nodename = "include"
    attribute_names = ["file"]
    children = []
    super(MjVisual, self).__init__(nodename, attribute_names, children)


# compiler
#TODO: implement


# option
#TODO: implement


# size
#TODO: implement


# visual

class MjVisual(MjBaseElement):
  """
  Python equivalent of MJCF visual.
  """

  def __init__(self):
    nodename = "visual"
    attribute_names = []
    children = []
    super(MjVisual, self).__init__(nodename, attribute_names, children)


class MjGlobal(MjBaseElement):
  """
  Python equivalent of MJCF global.
  """

  def __init__(self):
    nodename = "global"
    attribute_names = ["fovy", "ipd", "linewidth", "glow", "offwidth",
        "offheight"]
    children = []
    super(MjGlobal, self).__init__(nodename, attribute_names, children)


class MjQuality(MjBaseElement):
  """
  Python equivalent of MJCF quality.
  """

  def __init__(self):
    nodename = "quality"
    attribute_names = ["shadowsize", "offsamples", "numslices",
        "numstacks", "numarrows", "numquads"]
    children = []
    super(MjQuality, self).__init__(nodename, attribute_names, children)


class MjHeadlight(MjBaseElement):
  """
  Python equivalent of MJCF headlight.
  """

  def __init__(self):
    nodename = "headlight"
    attribute_names = ["ambient", "diffuse", "specular", "active"]
    children = []
    super(MjHeadlight, self).__init__(nodename, attribute_names, children)


class MjMap(MjBaseElement):
  """
  Python equivalent of MJCF map.
  """

  def __init__(self):
    nodename = "map"
    attribute_names = ["stiffness", "stiffnessrot", "force", "torque",
        "alpha", "fogstart", "fogend", "znear", "zfar", "shadowclip",
        "shadowscale"]
    children = []
    super(MjMap, self).__init__(nodename, attribute_names, children)


class MjScale(MjBaseElement):
  """
  Python equivalent of MJCF scale.
  """

  def __init__(self):
    nodename = "scale"
    attribute_names = ["forcewidth", "contactwidth", "contactheight",
        "connect", "com", "camera", "light", "selectpoint", 
        "jointlength", "jointwidth", "actuatorlength", 
        "actuatorwidth", "framelength", "framewidth", "constraint",
        "slidercrank"]
    children = []
    super(MjScale, self).__init__(nodename, attribute_names, children)


class MjRgba(MjBaseElement):
  """
  Python equivalent of MJCF rgba.
  """

  def __init__(self):
    nodename = "rgba"
    attribute_names = ["fog", "force", "inertia", "joint", "actuator",
        "com", "camera", "light", "selectpoint", "connect", 
        "contactpoint", "contactforce", "contactfriction", 
        "contacttorque", "constraint", "slidercrank", "crankbroken"]
    children = []
    super(MjRgba, self).__init__(nodename, attribute_names, children)


# statistic

class MjStatistic(MjBaseElement):
  """
  Python equivalent of MJCF statistic.
  """

  def __init__(self):
    nodename = "statistic"
    attribute_names = ["meaninertia", "meanmass", "meansize", "extent",
        "center"]
    children = []
    super(MjStatistic, self).__init__(nodename, attribute_names, children)


# default
#TODO: implement


# custom
#TODO: implement


# asset

class MjAsset(MjBaseElement):
  """
  Python equivalent of MJCF asset.
  """

  def __init__(self):
    nodename = "asset"
    attribute_names = []
    children = []
    super(MjAsset, self).__init__(nodename, attribute_names, children)


class MjTexture(MjBaseElement):
  """
  Python equivalent of MJCF texture.
  """

  def __init__(self):
    nodename = "texture"
    attribute_names = ["name", "type", "file", "gridsize", "gridlayout",
        "fileright", "fileleft", "fileup", "filedown", "filefront",
        "fileback", "builtin", "rgb1", "rgb2", "mark", "markrgb", "random",
        "width", "height"]
    children = []
    super(MjTexture, self).__init__(nodename, attribute_names, children)


class MjHfield(MjBaseElement):
  """
  Python equivalent of MJCF hfield.
  """

  def __init__(self):
    nodename = "hfield"
    attribute_names = ["name", "file", "nrow", "ncol", "size"]
    children = []
    super(MjHfield, self).__init__(nodename, attribute_names, children)


class MjMesh(MjBaseElement):
  """
  Python equivalent of MJCF mesh.
  """

  def __init__(self):
    nodename = "mesh"
    attribute_names = ["name", "class", "file", "scale"]
    children = []
    super(MjMesh, self).__init__(nodename, attribute_names, children)


class MjMaterial(MjBaseElement):
  """
  Python equivalent of MJCF material.
  """

  def __init__(self):
    nodename = "material"
    attribute_names = ["name", "class", "texture", "texrepeat", "texuniform",
        "emission", "specular", "shininess", "reflectance", "rgba"]
    children = []
    super(MjMaterial, self).__init__(nodename, attribute_names, children)


# (world)body

class MjWorldbody(MjBaseElement):
  """
  Python equivalent of MJCF worldbody.
  """

  def __init__(self):
    nodename = "worldbody"
    attribute_names = []
    children = []
    super(MjWorldbody, self).__init__(nodename, attribute_names, children)


class MjBody(MjBaseElement):
  """
  Python equivalent of MJCF body.
  """

  def __init__(self):
    nodename = "body"
    attribute_names = ["name", "childclass", "pos", "quat", "mocap", 
        "axisangle", "xyaxes", "zaxis", "euler", "user"]
    children = []
    super(MjBody, self).__init__(nodename, attribute_names, children)


class MjInertial(MjBaseElement):
  """
  Python equivalent of MJCF inertial.
  """

  def __init__(self):
    nodename = "inertial"
    attribute_names = ["pos", "quat", "mass", "diaginertia", "axisangle",
        "xyaxes", "zaxis", "euler", "fullinertia"]
    children = []
    super(MjInertial, self).__init__(nodename, attribute_names, children)


class MjJoint(MjBaseElement):
  """
  Python equivalent of MJCF joint.
  """

  def __init__(self):
    nodename = "joint"
    attribute_names = ["pos", "quat", "mass", "diaginertia", "axisangle",
        "xyaxes", "zaxis", "euler", "fullinertia"]
    children = []
    super(MjJoint, self).__init__(nodename, attribute_names, children)


class MjFreejoint(MjBaseElement):
  """
  Python equivalent of MJCF freejoint.
  """

  def __init__(self):
    nodename = "freejoint"
    attribute_names = ["name"]
    children = []
    super(MjFreejoint, self).__init__(nodename, attribute_names, children)


class MjGeom(MjBaseElement):
  """
  Python equivalent of MJCF geom.
  """

  def __init__(self):
    nodename = "geom"
    attribute_names = ["name", "class", "type", "contype", "conaffinity",
        "condim", "group", "size", "material", "friction", "mass", 
        "density", "solmix", "solref", "solimp", "margin", "gap", 
        "fromto", "pos", "quat", "axisangle", "xyaxes", "zaxis", "euler", 
        "hfield", "mesh", "fitscale", "rgba", "user"]
    children = []
    super(MjGeom, self).__init__(nodename, attribute_names, children)


class MjSite(MjBaseElement):
  """
  Python equivalent of MJCF site.
  """

  def __init__(self):
    nodename = "site"
    attribute_names = ["name", "class", "type", "group", "pos", "quat",
        "material", "size", "axisangle", "xyaxes", "zaxis", "euler", 
        "rgba", "user"]
    children = []
    super(MjSite, self).__init__(nodename, attribute_names, children)


class MjCamera(MjBaseElement):
  """
  Python equivalent of MJCF camera.
  """

  def __init__(self):
    nodename = "camera"
    attribute_names = ["name", "class", "fovy", "ipd", "pos", "quat",
        "axisangle", "xyaxes", "zaxis", "euler", "mode", "target", "user"]
    children = []
    super(MjCamera, self).__init__(nodename, attribute_names, children)


class MjLight(MjBaseElement):
  """
  Python equivalent of MJCF light.
  """

  def __init__(self):
    nodename = "light"
    attribute_names = ["name", "class", "directional", "castshadow",
        "active", "pos", "dir", "attenuation", "cutoff", "exponent", 
        "ambient", "diffuse", "specular", "mode", "target"]
    children = []
    super(MjLight, self).__init__(nodename, attribute_names, children)


# contact
#TODO: implement


# equality
#TODO: implement


# tendon
#TODO: implement


# actuator
#TODO: implement


# sensor

class MjSensor(MjBaseElement):
  """
  Python equivalent of MJCF sensor.
  """

  def __init__(self):
    nodename = "sensor"
    attribute_names = []
    children = []
    super(MjSensor, self).__init__(nodename, attribute_names, children)


class MjAccelerometer(MjBaseElement):
  """
  Python equivalent of MJCF accelerometer.
  """

  def __init__(self):
    nodename = "accelerometer"
    attribute_names = ["name", "site", "cutoff", "noise", "user"]
    children = []
    super(MjAccelerometer, self).__init__(nodename, attribute_names, children)


class MjVelocimeter(MjBaseElement):
  """
  Python equivalent of MJCF velocimeter.
  """

  def __init__(self):
    nodename = "velocimeter"
    attribute_names = ["name", "site", "cutoff", "noise", "user"]
    children = []
    super(MjVelocimeter, self).__init__(nodename, attribute_names, children)


# keyframe
#TODO: implement
