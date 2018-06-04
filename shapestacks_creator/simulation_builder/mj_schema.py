"""
Base classes for Python mappings of MuJoCo model objects according to the XML
schema definition for MuJoCo models (MJCF).
"""


class MjcfElement(object):
  """
  A generic MjcfElement which intstantiates a generic Python object from an
  MJCF compliant XML node.
  Overwrites standard Python object behaviour.
  """

  def __init__(self, nodename, attributes, children):
    _nodename = nodename
    _attributes = {}
    _children = []
    for att, val in attributes.items():
      if not (isinstance(att, str) and isinstance(val, str)):
        raise TypeError("Attribute names and values of MjcfElements can only be strings!")
      _attributes.update({att : val})
    for child in children:
      if not isinstance(child, MjcfElement):
        raise TypeError("Children of a MjcfElement must be a MjcfElement themselves!")
      _children.append(child)
    super(MjcfElement, self).__setattr__('_nodename', _nodename)
    super(MjcfElement, self).__setattr__('_attributes', _attributes)
    super(MjcfElement, self).__setattr__('_children', _children)

  def __getattr__(self, att):
    if att in self._attributes:
      return self._attributes[att]
    else:
      raise AttributeError("The MjcfElement does not have an attribute named '%s'!" % att)

  def __setattr__(self, att, val):
    if att in self._attributes:
      if isinstance(val, str):
        self._attributes[att] = val
      else:
        raise TypeError("Values of MjcfElement attributes can only be strings!")
    else:
      raise AttributeError("The MjcfElement does not have an attribute named '%s'!" % att)

  def element_type(self):
    """
    Returns the type of the MjcfElement (nodename).
    """
    return self._nodename

  #TODO: define proper print function


class MjcfFormat(object):
  """
  Helper class with static formatting functions for the MJCF format.
  """

  @staticmethod
  def tuple(tup):
    """
    Converts a tuple to a correctly formatted string.
    """
    return " ".join([str(t) for t in tup])

  @staticmethod
  def strtuple2float(strtup):
    """
    Converts a tuple string to a tuple of floats.
    """
    return tuple([float(t) for t in strtup.split(' ')])
