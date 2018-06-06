# MuJoCo Installation for ShapeStacks

1. Download and the [MuJoCo binaries for Linux](https://www.roboti.us/download/mjpro150_linux.zip) and unzip them to your home directory under ```${HOME}/.mujoco/```

2. Obtain a valid [MuJoCo license](https://www.roboti.us/license.html) and place the key file under ```${HOME}/.mujoco/mjkey.txt```

3. Add the MuJoCo files to your ```LD_LIBRARY_PATH``` to make the accessible during the mujoco-py compilation.

```bash
$ export MUJOCO_PATH=${HOME}/.mujoco/mjpro150
$ export LD_LIBRARY_PATH=${MUJOCO_PATH}/bin:${LD_LIBRARY_PATH}
```

4. OpenAI's [mujoco-py](https://github.com/openai/mujoco-py) wrapper requires an nvidia-384 driver. Install it, if you do not have it already and add the libraries to your ```LD_LIBRARY_PATH``` to make the accessible during the mujoco-py compilation.

```bash
$ sudo apt-get install nvidia-384 nvidia-384-dev
$ export LD_LIBRARY_PATH=/usr/lib/nvidia-384:${LD_LIBRARY_PATH}
```

5. MuJoCo and mujoco-py also need the following libraries for rendering purposes (both onscreen and offscreen). Install them, if you have not done already.

```bash
sudo apt-get install \
  libgl1-mesa-dev \
  libgl1-mesa-glx \
  libosmesa6-dev \
  libglew \
  libglew-dev \
  libglfw3-dev
```

6. mujoco-py also needs a custom patchelf library:

```bash
$ sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf
$ sudo chmod +x /usr/local/bin/patchelf
```

7. Activate the ShapeStacks virtual environment:

```bash
$ . ./activate_venv.sh
Set environment varibale SHAPESTACKS_CODE_HOME=/path/to/this/repository
Activated virtual environment 'venv'.
```

8. Install mujoco-py via pip3 into the ShapeStacks virtual environment.

```bash
(venv) $ pip3 install mujoco-py
```

9. Launch python3 in the ShapeStacks environment and import mujoco-py to invoke the compilation of the Cython wrappers (this only occurs on the first import of the library).

```bash
(venv) $ pip3 install mujoco-py
>>> import mujoco_py
```

10. In case any errors occur during the installation process of MuJoCo or mujoco-py, we refer to the websites of the original maintainers of those libraries for troubleshooting:
- [MuJoCo Forum](http://www.mujoco.org/forum/index.php)
- [mujoco-py on GitHub](https://github.com/openai/mujoco-py)
