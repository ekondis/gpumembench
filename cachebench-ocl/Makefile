# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

include ../common.mk
CC = g++
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=0  -funroll-loops -ffast-math -fassociative-math -msse2 -msse
INCDIR = ../common
FLAGS = ${OPTFLAG} -I${OPENCL_INSTALL_PATH}/include -Wall -g -I${INCDIR}
LFLAGS = -L${OPENCL_LIBRARY_PATH} -lm -lstdc++ -lOpenCL -lrt

cachebench-ocl: main.o cache_kernels.o
	${CC} -o $@ $^ ${LFLAGS}

main.o: main.cpp cache_kernels.h ${INCDIR}/loclutil.h
	${CC} -c ${FLAGS} $<

cache_kernels.o: cache_kernels.cpp kernel.h
	${CC} ${FLAGS} -c $<

kernel.h: kernel.cl
	echo "const char c_kernel[]={" >kernel.h
	hexdump -ve '1/1 "0x%.2x,"' kernel.cl >>kernel.h
	echo "0x00};" >>kernel.h

clean:
	\rm -f cachebench-ocl main.o cache_kernels.o kernel.h

rebuild: clean cachebench-ocl

