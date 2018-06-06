# Recording ShapeStacks Scenarios

After activating the ShapeStacks virtual environment, run the recording script as follows:

```bash
(venv) $ LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so python3 record_scenario.py \
  --mjmodel_path ${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf/env_ccs-easy-h=5-vcom=2-vpsf=0-v=1.xml \
  --record_path ${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/recordings/env_ccs-easy-h=5-vcom=2-vpsf=0-v=1 \
  --mjsim_time 4 \
  --fps 8 \
  --max_frames 32 \
  --cameras cam_1 cam_2 \
  --res 224 224 \
  --formats rgb
```

This will run the simulation forward from the beginning for 4 seconds and record RGB renderings with 8 FPS (up to 32 frames) at a resolution of 224 x 224 pixels.

After the maximum simulation time, the recording script will also report if the stack has collapsed (topmost shape has moved) or not.
