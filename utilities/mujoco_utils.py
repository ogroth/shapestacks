"""
Contains utility functions to work with a mujoco simulation.
"""

import mujoco_py

def mjsim_mat_id2name(sim: mujoco_py.MjSim):
  """
  Returns all material names and their corresponding IDs.
  """
  mat_ids = list(range(sim.model.nmat))
  mat_id2name = {}
  for mat_id in mat_ids:
    mat_name_addr_start = sim.model.name_matadr[mat_id]
    mat_name_addr_end = mat_name_addr_start
    char = sim.model.names[mat_name_addr_end].decode('UTF-8')
    while not char == '': # names are delineated by empty strings within names array
      mat_name_addr_end += 1
      char = sim.model.names[mat_name_addr_end].decode('UTF-8')
    mat_name = ''.join(list(
        map(lambda e: e.decode('UTF-8'),
            list(sim.model.names[mat_name_addr_start:mat_name_addr_end]))))
    mat_id2name.update({mat_id : mat_name})
  return mat_id2name

def mjhlp_geom_type_id2name(geom_type_id: int) -> str:
  geom_types = ['plane', 'hfield', 'sphere', 'capsule', 'ellipsoid', \
      'cylinder', 'box', 'mesh']
  return geom_types[geom_type_id]
