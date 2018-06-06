# Running the Stacking Simulation

Before you can run the stacker, you need a trained stability predictor. You can either train one yourself with the training script provided [train_inception_v4_shapestacks.py](../intuitive_physics/stability_predictor/train_inception_v4_shapestacks.py) or you can [download](http://shapestacks.robots.ox.ac.uk/static/download/v1/shapestacks-incpv4.tar.gz) the pre-trained models from the original [ShapeStacks paper](https://arxiv.org/pdf/1804.08018.pdf) and place them under ```${SHAPESTACKS_CODE_HOME}/models/```.

After activating the ShapeStacks virtual environment, run the stacking script as follows:

```bash
(venv) $ python3 run_stacker.py \
  --mjmodel_path ${SHAPESTACKS_CODE_HOME}/simulations/env-stack-ccs_simple.xml \
  --tfckpt_dir ${SHAPESTACKS_CODE_HOME}/models/shapestacks-incpv4/shapestacks-ccs/snapshots/real=0.663286 \
  --mjsim_dir /tmp/simulations/stack-ccs_simple \
  --mode stacking \
  --rendering_mode onscreen
```

This runs the stacking simulation with the loaded snapshot of the stability predictor and shows the procedure in an onscreen window. The results of the simulation (periodic screenshots and simulation state snapshots) are written to the directory specified by ```--mjsim_dir```.

If you prefer to run batch simulations in offscreen mode (```--rendering_mode offscreen```), you need to preload the correct OpenGL library with ```LD_PRELOAD=/usr/lib/nvidia-384/libOpenGL.so```. Also, if you save multiple runs into the same output directory, consider changing the ```--run_prfx``` flag to avoid overwriting previous results.

The stacking script can also be run in 'balancing' mode (```--mode balancing```) with one of the balancing simulations (```${SHAPESTACKS_CODE_HOME}/simulations/env-balance-*.xml```).
