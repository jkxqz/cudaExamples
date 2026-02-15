SUMMARY: 
vectorAddition.cu computes the vector sum of two matrices A and B, putting the result into a third vector C (all of length n).

DETAILS:
Beginning in main(), memory for the three matrices (A, B, and C) is allocated on the host machine. A and B are then filled with initial nonzero values. From main(), the length, n, and the addresses of the three matrices are passed to host function vecAdd() for GPU prep. This prep consists of ensuring that the GPU can receive copies of each of the arrays and launch the vecAddKernel() GPU kernel to run the parallelized computation on the device. Once the summation has been performed, the result vector C is then copied back to the host memory (overwriting any initialized values), then the GPU memory is deallocated. 
