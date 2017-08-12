# gpumembench benchmark suite

In this repository a GPU benchmark tool is hosted regarding the evaluation of on-chip GPU memories from a memory bandwidth perspective. In particular, 3 benchmark tools are provided for the assessment of L1-L2-texture caches, shared memory and constant memory cache, respectively.

CUDA and OpenCL implementations are provided.

To build this tools use the provided Makefile. If it is needed you have to set the CUDA_INSTALL_PATH, OPENCL_INSTALL_PATH & OPENCL_LIBRARY_PATH variables in "common.mk" to point to the proper CUDA/OpenCL directories.

Execution results
--------------

Here some indicative output results of executions on a GTX-480 are provided:

An extract of the cachebench output follows:
```
CUDA cachebench (repeated memory cached operations microbenchmark)
------------------------ Device specifications ------------------------
Device:              GeForce GTX 480
CUDA driver version: 8.0
GPU clock rate:      1550 MHz
Memory clock rate:   950 MHz
Memory bus width:    384 bits
WarpSize:            32
L2 cache size:       768 KB
Total global mem:    1530 MB
ECC enabled:         No
Compute Capability:  2.0
Total SPs:           480 (15 MPs x 32 SPs/MP)
Compute throughput:  1488.00 GFlops (theoretical single precision FMAs)
Memory bandwidth:    182.40 GB/sec
-----------------------------------------------------------------------
Total GPU memory 1605042176, free 1481957376
Buffer size: 512MB
Whole cache hierarchy benchmark (L1 & L2 caches)

Read only benchmark
EXCEL header:
Element size,Grid size, Parameters,   , Data size,        ,Execution time,Instr.thr/put,Memory b/w, Ops/sec,Ops/cycle
     (bytes),(threads),(step),(idx/cl),(elements), (bytes),       (msecs),      (GIOPS),  (GB/sec),  (10^9),   per SM
           4,    23040,     1,     512,       512,    2048,         0.624,      302.707,  1210.827, 302.707,   13.020
           4,    23040,     1,    1024,      1024,    4096,         0.549,      343.840,  1375.362, 343.840,   14.789
           4,    23040,     1,    2048,      2048,    8192,         0.549,      343.840,  1375.362, 343.840,   14.789
           4,    23040,     1,    4096,      4096,   16384,         0.549,      343.820,  1375.282, 343.820,   14.788
           4,    23040,     1,    8192,      8192,   32768,         0.548,      344.141,  1376.566, 344.141,   14.802
           4,    23040,     1,       0,     23040,   92160,         0.547,      345.027,  1380.109, 345.027,   14.840
           4,    23040,     2,       0,     46080,  184320,         0.684,      275.851,  1103.403, 275.851,   11.865
           4,    23040,     3,       0,     69120,  276480,         0.949,      198.902,   795.608, 198.902,    8.555
           4,    23040,     4,       0,     92160,  368640,         0.960,      196.641,   786.563, 196.641,    8.458
           4,    23040,     5,       0,    115200,  460800,         1.993,       94.725,   378.900,  94.725,    4.074
           4,    23040,     6,       0,    138240,  552960,         2.758,       68.444,   273.776,  68.444,    2.944
           4,    23040,     7,       0,    161280,  645120,         2.793,       67.583,   270.332,  67.583,    2.907
           4,    23040,     8,       0,    184320,  737280,         2.875,       65.648,   262.590,  65.648,    2.824
           4,    23040,     9,       0,    207360,  829440,         3.107,       60.751,   243.006,  60.751,    2.613
           4,    23040,    10,       0,    230400,  921600,         3.316,       56.924,   227.696,  56.924,    2.448
           4,    23040,    11,       0,    253440, 1013760,         3.665,       51.501,   206.005,  51.501,    2.215
           4,    23040,    12,       0,    276480, 1105920,         4.146,       45.524,   182.096,  45.524,    1.958
           4,    23040,    13,       0,    299520, 1198080,         4.795,       39.364,   157.454,  39.364,    1.693
           4,    23040,    14,       0,    322560, 1290240,         4.455,       42.370,   169.481,  42.370,    1.822
           4,    23040,    15,       0,    345600, 1382400,         4.634,       40.733,   162.933,  40.733,    1.752
           4,    23040,    16,       0,    368640, 1474560,         4.979,       37.906,   151.624,  37.906,    1.630
           4,    23040,    18,       0,    414720, 1658880,         4.761,       39.646,   158.585,  39.646,    1.705
           4,    23040,    20,       0,    460800, 1843200,         4.626,       40.802,   163.206,  40.802,    1.755
           4,    23040,    22,       0,    506880, 2027520,         4.888,       38.615,   154.460,  38.615,    1.661
           4,    23040,    24,       0,    552960, 2211840,         4.882,       38.660,   154.641,  38.660,    1.663
           4,    23040,    28,       0,    645120, 2580480,         4.508,       41.871,   167.483,  41.871,    1.801
           4,    23040,    32,       0,    737280, 2949120,         4.883,       38.657,   154.628,  38.657,    1.663
           4,    23040,    40,       0,    921600, 3686400,         4.863,       38.813,   155.253,  38.813,    1.669
           4,    23040,    48,       0,   1105920, 4423680,         4.864,       38.807,   155.229,  38.807,    1.669
           4,    23040,    56,       0,   1290240, 5160960,         4.513,       41.818,   167.273,  41.818,    1.799
           4,    23040,    64,       0,   1474560, 5898240,         4.879,       38.684,   154.735,  38.684,    1.664
...

Peak bandwidth measurements per element size and access type
	Read only accesses:
		int1:    1380.11 GB/sec
		int2:    1479.92 GB/sec
		int4:    1458.74 GB/sec
		max:     1479.92 GB/sec
	Read-write accesses:
		int1:     423.37 GB/sec
		int2:     419.64 GB/sec
		int4:     342.76 GB/sec
		max:      423.37 GB/sec
```


shmembench execution output:
```
CUDA shmembench (shared memory bandwidth microbenchmark)
------------------------ Device specifications ------------------------
Device:              GeForce GTX 480
CUDA driver version: 8.0
GPU clock rate:      1550 MHz
Memory clock rate:   950 MHz
Memory bus width:    384 bits
WarpSize:            32
L2 cache size:       768 KB
Total global mem:    1530 MB
ECC enabled:         No
Compute Capability:  2.0
Total SPs:           480 (15 MPs x 32 SPs/MP)
Compute throughput:  1488.00 GFlops (theoretical single precision FMAs)
Memory bandwidth:    182.40 GB/sec
-----------------------------------------------------------------------
Total GPU memory 1605042176, free 1481957376
Buffer sizes: 3x8MB
Kernel execution time
	benchmark_shmem  (32bit):    57.964 msecs
	benchmark_shmem  (64bit):    57.943 msecs
	benchmark_shmem (128bit):    87.491 msecs
Total operations executed
	shared memory traffic    :          86 GB
	shared memory operations : 21487419392 operations (32bit)
	shared memory operations : 10743709696 operations (64bit)
	shared memory operations :  5371854848 operations (128bit)
Memory throughput
	using  32bit operations   : 1482.81 GB/sec (370.70 billion accesses/sec)
	using  64bit operations   : 1483.35 GB/sec (185.42 billion accesses/sec)
	using 128bit operations   :  982.38 GB/sec ( 61.40 billion accesses/sec)
	peak operation throughput :  370.70 Giga ops/sec
Normalized per SM
	shared memory operations per clock (32bit) :  239.16 (per SM 15.94)
	shared memory operations per clock (64bit) :  119.63 (per SM  7.98)
	shared memory operations per clock (128bit):   39.61 (per SM  2.64)
```

constbench execution output:
```
constbench (constant memory bandwidth microbenchmark)
------------------------ Device specifications ------------------------
Device:              GeForce GTX 480
CUDA driver version: 8.0
GPU clock rate:      1550 MHz
Memory clock rate:   950 MHz
Memory bus width:    384 bits
WarpSize:            32
L2 cache size:       768 KB
Total global mem:    1530 MB
ECC enabled:         No
Compute Capability:  2.0
Total SPs:           480 (15 MPs x 32 SPs/MP)
Compute throughput:  1488.00 GFlops (theoretical single precision FMAs)
Memory bandwidth:    182.40 GB/sec
-----------------------------------------------------------------------
Total GPU memory 1605042176, free 1480908800
Kernel execution time
	benchmark_constant  (32bit):   12.3310 msecs
	benchmark_constant  (64bit):    7.9485 msecs
	benchmark_constant (128bit):    9.9482 msecs
Total operations executed
	constant memory array size :        4096 bytes
	constant memory traffic    :       17180 MB
	constant memory operations :  4294967296 operations (32bit)
	constant memory operations :  2147483648 operations (64bit)
	constant memory operations :  1073741824 operations (128bit)
Memory throughput
	using  32bit operations : 1393.23 GB/sec (348.31 billion accesses/sec)
	using  64bit operations : 2161.40 GB/sec (270.18 billion accesses/sec)
	using 128bit operations : 1726.93 GB/sec (107.93 billion accesses/sec)
Normalized per SM
	Constant memory operations per clock (32bit) :  224.71 (per SM 14.98)
	Constant memory operations per clock (64bit) :  174.31 (per SM 11.62)
	Constant memory operations per clock (128bit):   69.63 (per SM  4.64)
Compute overhead
	Addition operations per constant memory operation  (32bit): 1
	Addition operations per constant memory operation  (64bit): 2
	Addition operations per constant memory operation (128bit): 4
```

Publications
--------------

If you find this benchmark tool useful for your research please don't forget to provide citation to the following paper:

*Konstantinidis, E.; Cotronis, Y., "A quantitative performance evaluation of fast on-chip memories of GPUs",* 
*24th Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP), Heraklion, Crete, Greece, pp. 448-455, 2016*  
*doi: 10.1109/PDP.2016.56*
