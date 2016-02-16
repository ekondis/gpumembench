# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

CUDA_INSTALL_PATH = /usr/local/cuda
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
INCDIR = -I../common
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall -g ${INCDIR}
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -g -ftz=true ${INCDIR}
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_30,code=\"compute_30,sm_30\"

.PHONY: all

all: cachebench cachebench-l2-only cachebench-tex-loads

cachebench: main.o cache_kernels.o
	${CC} -o $@ $^ ${LFLAGS}

cachebench-l2-only: main.o cache_kernels_l2.o
	${CC} -o $@ $^ ${LFLAGS}

cachebench-tex-loads: main.o cache_kernels_texloads.o
	${CC} -o $@ $^ ${LFLAGS}

main.o: main.cpp cache_kernels.h
	${CC} -c ${FLAGS} $<

cache_kernels.o: cache_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

cache_kernels_l2.o: cache_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -DL2_ONLY -c $< -o $@

cache_kernels_texloads.o: cache_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -DTEX_LOADS -c $< -o $@

clean:
	\rm -f cachebench cachebench-l2-only cachebench-tex-loads main.o cache_kernels.o cache_kernels_l2.o cache_kernels_texloads.o

rebuild: clean all

