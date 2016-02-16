# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

CUDA_INSTALL_PATH = /usr/local/cuda
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -ffast-math -fassociative-math -msse2 -msse
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_20,code=\"compute_20\" -ftz=true

constbench: main.o const_kernels.o
	${CC} -o $@ $^ ${LFLAGS}

main.o: main.cpp const_kernels.h
	${CC} -c ${FLAGS} $<

const_kernels.o: const_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $<

clean:
	\rm -f constbench main.o const_kernels.o

rebuild: clean constbench

