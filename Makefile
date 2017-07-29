# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

.PHONY: all clean rebuild

all:
	$(MAKE) -C cachebench-cuda
	$(MAKE) -C shmembench-cuda
	$(MAKE) -C shmembench-ocl
	$(MAKE) -C constbench-cuda
	mkdir -p bin/cuda bin/opencl
	cp cachebench-cuda/cachebench cachebench-cuda/cachebench-l2-only cachebench-cuda/cachebench-tex-loads shmembench-cuda/shmembench constbench-cuda/constbench bin/cuda
	cp shmembench-cuda/shmembench bin/opencl

clean:
	$(MAKE) -C cachebench-cuda clean
	$(MAKE) -C shmembench-cuda clean
	$(MAKE) -C shmembench-ocl clean
	$(MAKE) -C constbench-cuda clean

rebuild:
	$(MAKE) -C cachebench-cuda rebuild
	$(MAKE) -C shmembench-cuda rebuild
	$(MAKE) -C shmembench-ocl rebuild
	$(MAKE) -C constbench-cuda rebuild
