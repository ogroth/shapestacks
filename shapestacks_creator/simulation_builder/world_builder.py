"""
Contains builder classes to create model world files programmatically.
"""

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from simulation_builder.mj_schema import MjcfFormat
from simulation_builder.mj_elements import MjWorldbody, MjBody, MjFreejoint, \
    MjGeom, MjSensor, MjBaseElement, MjVelocimeter, MjSite, MjCamera, \
    MjLight


class MjWorldBuilder(object):
  """
  A world builder for MuJoCo worlds. Can load / create a MuJoCo world, insert
  objects into the worldspace and export it as MJCF compatible XML.
  """
  TL_TAG_WORLD = "world"

  def __init__(self):
    self._world = None
    self._worldbody = None
    self._sensor = None


  # world import / export

  def load_world(self, world_xml_file):
    """
    Loads a worldbody from a MuJoCo XML file.
    """
    self._world = ET.parse(world_xml_file).getroot()
    self._worldbody = self._world.find('worldbody')
    self._sensor = self._world.find('sensor')

  def new_world(self):
    """
    Creates a new empty world.
    """
    self._world = ET.Element(self.TL_TAG_WORLD)
    self._worldbody = ET.SubElement(self._world, MjWorldbody().to_etree_elem())
    self._sensor = ET.SubElement(self._world, MjSensor().to_etree_elem())

  def export_world(self, world_xml_file):
    """
    Exports the current worldbody to the specified file.
    """
    with open(world_xml_file, 'w') as f:
      xml_str = BeautifulSoup(ET.tostring(self._world), 'xml').prettify()
      f.write(xml_str)


  # basic world creation: static and dynamic geoms and meshes

  def insert_static(self, geom: MjGeom):
    """
    Inserts a static object into the world body which defines the static
    world space.
    """
    geom_elem = geom.to_etree_elem()
    self._worldbody.append(geom_elem)

  def insert_dynamic(self, geom: MjGeom):
    """
    Inserts a dynamic rigid object (freejoint body) into the world which can
    move within the static world space.
    """
    body = MjBody()
    body.pos = geom.pos
    body.euler = geom.euler
    geom.pos = MjcfFormat.tuple((0, 0, 0))
    geom.euler = MjcfFormat.tuple((0, 0, 0))
    freejoint = MjFreejoint()
    body.add_child_elem(freejoint)
    body.add_child_elem(geom)
    body_elem = body.to_etree_elem()
    self._worldbody.append(body_elem)


  # internal helper functions

  def _init_sensors(self):
    """
    Creates a new sensor list.
    """
    self._world.append(MjSensor().to_etree_elem())
    self._sensor = self._world.find('sensor')


  # helper functions

  def attach_site(self, site, geom_name, site_pos=None):
    """
    Statically attaches a site to a geom specified by 'geom_name'.
    Default attachment position is the geom center if not specified otherwise.
    """
    geom_body_etree_elem = self.find_geom_body(geom_name)
    if geom_body_etree_elem is None:
      raise ValueError("The worldbody does not contain the geom '%s' to attach the site to!" % geom_name)
    geom_body = MjBaseElement.from_etree_elem(geom_body_etree_elem)
    site_body = MjBody()
    if site_pos is None:
      site.pos = [c for c in geom_body.get_children() if c.name == geom_name][0].pos
    else:
      site.pos = site_pos
    site_body.add_child_elem(site)
    geom_body_etree_elem.append(site_body.to_etree_elem())


  # world inspection: look up geoms and meshes

  def find_geom_body(self, geom_name):
    """
    Searches the worldbody for a geom of 'geom_name' and returns a pointer
    to the enclosing body node (xml.etree.Element).
    Returns 'None' if no matching geom can be found.
    """
    xpath_string = ".//geom[@name='%s'].." % geom_name
    return self._worldbody.find(xpath_string)


  # instrumentation: attachment of lights, cameras and sensors

  def insert_static_light(self, light):
    """
    Inserts a static light into the worldbody.
    """
    if not isinstance(light, MjLight):
      raise TypeError("Inserted light must be a MjLight object!")
    light.mode = "fixed"
    light_elem = light.to_etree_elem()
    self._worldbody.append(light_elem)

  def insert_static_camera(self, camera):
    """
    Inserts a static camera into the worldbody.
    """
    if not isinstance(camera, MjCamera):
      raise TypeError("Inserted camera must be a MjCamera object!")
    camera.mode = "fixed"
    camera_elem = camera.to_etree_elem()
    self._worldbody.append(camera_elem)

  def attach_sensor(self, sensor, geom_name):
    """
    Attaches a sensor to the geom specified by 'geom_name'.
    Creates an internal sensor binding site named 'sensor_site_%s' % sensor.name
    depending on the default sensor requirements
    """
    sensor_site = MjSite()
    sensor_site.name = "sensor_site_%s" % sensor.name
    if isinstance(sensor, MjVelocimeter):
      sensor_site.size = "0.01"
    else:
      raise ValueError("Unsupported sensor type %s!" % type(sensor))
    self.attach_site(sensor_site, geom_name)
    sensor.site = sensor_site.name
    if self._sensor is None:
      self._init_sensors()
    self._sensor.append(sensor.to_etree_elem())
