# objective-guided-diffusion
This is a repository for the collaborative work between Chase DS and UCL


# How to install

Create a venv and install the requirements from the requirements.txt file.
We require `black` for code formatting.

```bash

Then, it is important to run the following commands to install the other dependencies (and which cannot be added to the requirements file directly):

```bash
pip install cupy==13.2.0
pip install git+https://github.com/tgcsaba/ksig.git --no-deps
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
There is a dependency at the moment on `cupy`, which requires a `CUDA` installation/VS build tools. You will get errors when running the code without it.

There is a thin balance to find between the packages for signatures, `torch` and `cupy`.
Here I go with the last release of Python `3.10`.

Furthermore, in order to fasttrack my own coding, I used a shortcut relying on the library `corai`. I coded it myself
but certain pieces are a bit deprecated.  
Normally, if you follow the following steps, you should have no issues installing it:

```bash
pip install corai>=1.4.01 --no-deps
pip install scipy==1.7.2
pip install networkx==3.3
```

I will fix that as soon as I can! So we do not rely on `corai` and struggle doing the installation.

# How to run

Go to `tests/<name_exp>/train_script.py`. This should be runnable in the usual manner, without command line arguments.

# Loggers
Change last line in `src/logger/config_logging.json`: `"level": "INFO"` to get more logs.