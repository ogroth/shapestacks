# Creating a new ShapeStacks Scenario

After activating the ShapeStacks virtual environment, run the scenario creation script as follows:

```bash
(venv) $ python3 create_scenario.py \
  --mjmodel_name ccs-easy-h=4-vcom=2-vpsf=0-v=1 \
  --template_path ${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf_template \
  --export_path ${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf \
  --shapes cuboid cylinder sphere \
  --height 4 \
  --vcom 2 \
  --unique_colors
```

This will create a new ShapeStack scenario under ```${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf``` based on the template world defined in ```${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf_template```.

The ```--mjmodel_name``` parameter needs to comply with the conventions set by the ShapeStacks dataset documentation as it encodes the stack's annotations and will be parsed during scene rendering and data loading.

The other parameters for stack creation (shapes, height, vcom, vpsf, etc.) are documented in [create_scenario.py](create_scenario.py) and can be set accordingly.

The resulting scenario created in ```${SHAPESTACKS_CODE_HOME}/data/shapestacks_example/mjcf``` can be executed by running its head environment XML file (```env_<mjmodel_name>.xml```) with MuJoCo.
