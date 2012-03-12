=====================================================================================
INPAR 2012: Machine Learning for Predictive Auto-tuning with Boosted Regression Trees
=====================================================================================

This was originally a fork of fbconv-gcg.

The latex source is in

* paper/final.tex for the camera-ready

* paper/draft1.tex for the original submission

The code in the root directory was used to run the experiments for the paper.

# TODO: clean this up

Dependencies:
=============

- pycuda
# if your compiler is > gcc-4.4, you'll have to use the dev version of
# pycuda, see "Known issues" section bellow and the following commit for more info:
# https://github.com/inducer/pycuda/commit/2b05f35784dc84033b15c07bcfbe8906de8d8cea
$ (cd /tmp && git clone https://github.com/inducer/pycuda.git && cd pycuda && git submodule init && git submodule update && python setup.py install)

- Cheetah
$ pip install -vUI Cheetah

or use your distro's package manager, e.g.:
$ emerge dev-python/cheetah


# TODO: replace Cheetah with Tempita to remove dependency ?
# http://pythonpaste.org/tempita/


Example of use:
===============

# create directory to store results
mkdir -p output_results

# lunch experiments
python fbconv3_benchmark.py output_results fbconv3_inputs_cherrypick.py fbconv3_cuda_metaparams_cherrypick.py


Known issues:
=============

- if you compiler is > gcc-4.4, you'll have to use
PYCUDA_NVCC_DEFAULT_FLAGS to specify another compiler for NVCC to use:
http://documen.tician.de/pycuda/driver.html#pycuda.compiler.DEFAULT_NVCC_FLAGS

e.g.:
$ export PYCUDA_DEFAULT_NVCC_FLAGS="--compiler-bindir=$(gcc-config -B x86_64-pc-linux-gnu-4.4.5)"

Note that in our experiments, we used additional flags:
$ export PYCUDA_DEFAULT_NVCC_FLAGS="--compiler-bindir=$(gcc-config -B x86_64-pc-linux-gnu-4.4.5) --opencc-options=-v,-OPT:0limit=0,-O3,-LIST:source=on --ptxas-options=-v"

