SUMMARY:
Convolve mxn matrix A with kxk filter F to produce mxn matrix B.

DETAILS:
'convolution.cu' has several implementations of GPU kernels that perform convolution on a 2-dimensional matrix A. Difference in implementations stems from (1) how threads are organized and (2) whether or not threads in a block cooperate with one another:
    case0: For 'basicConvKernel', there is exactly one thread for every element in matrix B, with no collaboration between threads (slowest case)
    case1: In 'tiledHaloConvKernelT2In', there are as many threads as necessary to create the output cells in the convolved matrix, implying a 'halo' of threads that do not update the output matrix, but simply serve to collaboratively load input values into shared memory for lower latency access.
    case2: TODO

Block size is hardcoded to 16 threads per x- and y-dimension; the z-dimension for block size is hardcoded to 1 as the code assumes a maximum of 2 dimensions needed to represent any matrix.

The convolution filter, F, is accessed by the GPU threads via constant global memory variable F_c. Constant memory data may be aggressively cached by the hardware since it's read-only from within a kernel, reducing strain on global memory bandwidth.

FILTER_RADIUS set as symbolic constant at top of file.

