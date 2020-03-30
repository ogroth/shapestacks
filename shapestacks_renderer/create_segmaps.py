"""Script to transform raw violation segmentation renderings into .map files."""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import cluster

sys.path.insert(0, os.environ['SHAPESTACKS_CODE_HOME'])
from utilities.mujoco_utils import mjsim_mat_id2name
from shapestacks_renderer.rendering_constants import VSEG_COLOR_CODES,  ISEG_COLOR_CODES


# ---------- command line arguments ----------

ARGPARSER = argparse.ArgumentParser(
    description='Preprocess the raw ShapeStacks segmentation recording data.')
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/shapestacks',
    help="The directory of the fairblocks data.")
ARGPARSER.add_argument(
    '--mask_res', type=int, nargs=2, default=[224, 224],
    help="The resolution of the segmentation map (height, width).")
ARGPARSER.add_argument(
    '--segtype', type=str, default='vseg',
    help="The segmentation type to convert. Available types are: vseg | iseg.")
# ARGPARSER.add_argument(
#     '--cleanup', action='store_true', default=False,
#     help="If set, the script will delete the original RGB segmentation maps.")
ARGPARSER.add_argument(
    '--recompute', action='store_true', default=False,
    help="Recompute a map, if already present.")


# ---------- constants ----------

COLOR_CODES = {
    'vseg' : VSEG_COLOR_CODES,
    'iseg' : ISEG_COLOR_CODES,
}
MAX_COLOR_SCALE = 0.3  # max. color scale of segmentation renderings
CHANNELS = 3
LABEL_RANGE = 256  # label maps are stored as PNG-compressed grayscale images


# ---------- main ----------

if __name__ == '__main__':
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("FLAGS: ", FLAGS)

  # directory setup
  recordings_dir = os.path.join(FLAGS.data_dir, 'recordings')

  # shape setup
  height = FLAGS.mask_res[0]
  width = FLAGS.mask_res[1]

  # loop over all recordings and transform all segmentation renderings into
  # sparse label arrays
  color_codes = COLOR_CODES[FLAGS.segtype]
  color_codes = [c[:-1] for c in color_codes]
  code_book = np.array(color_codes) * MAX_COLOR_SCALE
  rec_dirs = os.listdir(recordings_dir)
  file_prfx = '%s-' % (FLAGS.segtype, )
  for i, scn in enumerate(rec_dirs):
    scn_dir = os.path.join(recordings_dir, scn)
    print("%s / %s : %s" % (i+1, len(rec_dirs), scn_dir))
    for seg_img_file in filter(
        lambda f: f.startswith(file_prfx) and f.endswith('.png'), os.listdir(scn_dir)):

      # full size PNG segmentation map without rendering artifacts
      seg_map = None
      seg_map_file = seg_img_file.replace('.png', '.map')
      if not os.path.exists(os.path.join(scn_dir, seg_map_file)) or FLAGS.recompute:
        seg_img = plt.imread(os.path.join(scn_dir, seg_img_file))
        seg_map, dist = cluster.vq.vq(seg_img.reshape(height * width, CHANNELS), code_book)
        plt.imsave(
            os.path.join(scn_dir, seg_map_file),
            seg_map.reshape(height, width),
            vmin=0, vmax=len(color_codes), cmap='gray', format='png')
      # if FLAGS.cleanup:
      #   pass
