from data_provider.shapestacks_utils import extract_objects
worldfile_path = 'data/shapestacks_example/mjcf/world_ccs-easy-h=5-vcom=2-vpsf=0-v=1.xml'
shapestacks_objects = extract_objects(worldfile_path)
for obj in shapestacks_objects:
  print(obj)
