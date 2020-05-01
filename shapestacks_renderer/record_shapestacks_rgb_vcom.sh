#!/bin/sh
echo "Recording ShapeStacks scenarios from MuJoCo."

# filter parameters
# $1: dataset name
# $2: height filter

# paths
DATASET_NAME=$1
DATASET_ROOT_DIR="${SHAPESTACKS_CODE_HOME}/data/${DATASET_NAME}"
MJCF_ROOT_DIR="${DATASET_ROOT_DIR}/mjcf"
RECORD_ROOT_DIR="${DATASET_ROOT_DIR}/recordings"

# file filter
FILTER="env_.*-h=[$2]-vcom=[^0]-vpsf=0"

# recording options
# TIME=4
TIME=5
FPS=8
# MAX_FRAMES=1
MAX_FRAMES=40
RES="224 224"
# CAMERAS="cam_1 cam_2 cam_3 \
#     cam_4 cam_5 cam_6 \
#     cam_7 cam_8 cam_9 \
#     cam_10 cam_11 cam_12 \
#     cam_13 cam_14 cam_15 cam_16"
CAMERAS="cam_1 \
    cam_4 \
    cam_7 \
    cam_10"
FORMAT="rgb"
MAX_LIGHTS=5
MAX_WALLTEX=6
MAX_FLOORTEX=6
COLOR_MODE="original"

# helper functions
create_params()
{
  model_path=$1   # $1: model_path
  record_path=$2  # $2: record_path

  # randomize light and textures
  lightid=`python3 -c "import random; print(random.randint(0,${MAX_LIGHTS}-1))"`
  walltex=`python3 -c "import random; print(random.randint(0,${MAX_WALLTEX}-1))"`
  floortex=`python3 -c "import random; print(random.randint(0,${MAX_FLOORTEX}-1))"`
  # echo $lightid $walltex $floortex

  echo "--mjmodel_path ${model_path} --record_path ${record_path} \
      --mjsim_time ${TIME} --fps ${FPS} --max_frames ${MAX_FRAMES} \
      --res ${RES} --cameras ${CAMERAS} \
      --formats ${FORMAT} \
      --lightid ${lightid} --walltex ${walltex} --floortex ${floortex} \
      --color_mode ${COLOR_MODE}"
}

###
# Main body
###

# directory setup
mkdir ${DATASET_ROOT_DIR}
mkdir ${RECORD_ROOT_DIR}

# main loop over all simulation environments to record
for env_file in `ls ${MJCF_ROOT_DIR} | grep env_ | grep ${FILTER}`; do
  date
  echo "Recording ${env_file%".xml"} ..."

  # set up directory
  record_dir=${RECORD_ROOT_DIR}/${env_file%".xml"}
  log_file=${record_dir}/log.txt
  mkdir ${record_dir}

  # create params and render
  params=$(create_params "${MJCF_ROOT_DIR}/$env_file" $record_dir)
  # echo $params
  # LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so python3 record_scenario.py ${params} > ${log_file}
  LD_PRELOAD=/usr/lib/nvidia-418/libOpenGL.so python3 record_scenario.py ${params} > ${log_file}
done
