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
```
There is a dependency at the moment on `cupy`, which requires a `CUDA` installation/VS build tools. You will get errors when running the code without it.

There is a thin balance to find between the packages for signatures, `torch` and `cupy`.
Here I go with the last release of Python `3.10`.