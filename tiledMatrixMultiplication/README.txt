SUMMARY:
Compute result of matrix A (mxn) right multiplied by matrix B (nxk) to product matrix C (mxk). CUDA kernel makes use of tiling optimization.

DETAILS:
The tiling optimization used in 'tiledMatrixMultiplication.cu' for computing the matrix product works by reducing the total number of loads/stores from device global memory. This reduction in global memory accesses is accomplished by having the threads within a block collaborate via a set of shared memory data structures to eliminate redundant reads that would happen if individual threads were to work in isolation.

The matrix multiplication implementation in the the code creates one thread for each element in the output C matrix.
