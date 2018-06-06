"""
Contains simplified templates for MuJoCo objects which can be inserted into a
worldbody.
"""

from simulation_builder.mj_schema import MjcfFormat
from simulation_builder.mj_elements import MjGeom, MjLight


# abstract objects

class MjPrimitiveGeom(MjGeom):
  """
  An abstract primitive geometric shape as supported by MuJoCo.
  """

  DEFAULT_CONDIM = 6                          # complete friction and counter-forces
  DEFAULT_RGBA = (1.0, 1.0, 1.0, 1.0)         # white

  def __init__(self):
    super(MjPrimitiveGeom, self).__init__()
    self.condim = str(self.DEFAULT_CONDIM)
    self.rgba = MjcfFormat.tuple(self.DEFAULT_RGBA)

  def set_rgba(self, r, g, b, a):
    """
    Set the color of the geom. RGBA ranges in [0.0, 1.0].
    """
    t_rgba = (float(r), float(g), float(b), float(a))
    self.rgba = MjcfFormat.tuple(t_rgba)

  def set_pos(self, x, y, z):
    """
    Set the position of the geom.
    """
    t_pos = (float(x), float(y), float(z))
    self.pos = MjcfFormat.tuple(t_pos)

  def set_axisangle(self, x, y, z, a):
    """
    Set the rotation axis (x, y, y) and angle a.
    """
    t_axisangle = (float(x), float(y), float(z), float(a))
    self.axisangle = MjcfFormat.tuple(t_axisangle)


# primitive geometric shapes

class MjCuboid(MjPrimitiveGeom):
  """
  Simple cuboid.
  """

  DEFAULT_SIZE = (0.5, 0.5, 0.5)  # (x,y,z): dice with side length 1 (mj requires half-lengths)

  def __init__(self):
    super(MjCuboid, self).__init__()
    self.type = "box"
    self.size = MjcfFormat.tuple(self.DEFAULT_SIZE)

  def set_size(self, x, y, z):
    """
    Set the size of the cuboid with respect to its initial coordinate frame
    in x, y, z.
    """
    t_size = (float(x) / 2, float(y) / 2, float(z) / 2)
    self.size = MjcfFormat.tuple(t_size)

  def get_size(self):
    """
    Get the size of the cuboid as (x, y, z).
    """
    t_size = tuple([float(e) * 2 for e in self.size.split(" ")])
    return t_size


class MjCylinder(MjPrimitiveGeom):
  """
  Simple cylinder.
  """

  DEFAULT_SIZE = (0.5, 0.5)   # (r, h): radius, half-height
  DEFAULT_FRICTION = (1, 0.01, 0.01)    # sliding, torsional rolling

  def __init__(self):
    super(MjCylinder, self).__init__()
    self.type = "cylinder"
    self.size = MjcfFormat.tuple(self.DEFAULT_SIZE)
    self.friction = MjcfFormat.tuple(self.DEFAULT_FRICTION)

  def set_size(self, r, h):
    """
    Set the size of the cylinder as radius and height.
    """
    t_size = (float(r), float(h) / 2)
    self.size = MjcfFormat.tuple(t_size)

  def get_size(self):
    """
    Get the size of the cylinder as (r, h).
    """
    t_size = tuple([float(e) for e in self.size.split(" ")])
    t_size = (t_size[0], t_size[1] * 2)
    return t_size


class MjSphere(MjPrimitiveGeom):
  """
  Simple sphere.
  """

  DEFAULT_SIZE = (0.5, )   # (r, ): radius

  def __init__(self):
    super(MjSphere, self).__init__()
    self.type = "sphere"
    self.size = MjcfFormat.tuple(self.DEFAULT_SIZE)

  def set_size(self, r):
    """
    Set the size of the sphere via radius.
    """
    t_size = (float(r), )
    self.size = MjcfFormat.tuple(t_size)

  def get_size(self):
    """
    Get the size of the sphere as (r, ).
    """
    t_size = tuple([float(e) for e in self.size.split(" ")])
    return t_size


# light presets

class MjCameraHeadlight(MjLight):
  """
  A static light emulating the headlight behaviour of the free camera. Can be
  attached to a static camera for rendering purposes.
  """

  DEFAULT_DIRECTIONAL = "true"
  DEFAULT_AMBIENT = (0.1, 0.1, 0.1)
  DEFAULT_DIFFUSE = (0.4, 0.4, 0.4)
  DEFAULT_SPECULAR = (0.5, 0.5, 0.5)

  def __init__(self):
    super(MjCameraHeadlight, self).__init__()
    self.directional = self.DEFAULT_DIRECTIONAL
    self.ambient = MjcfFormat.tuple(self.DEFAULT_AMBIENT)
    self.diffuse = MjcfFormat.tuple(self.DEFAULT_DIFFUSE)
    self.specular = MjcfFormat.tuple(self.DEFAULT_SPECULAR)
