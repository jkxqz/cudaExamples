SUMMARY:
Right multiply mxn matrix A by nxk matrix B to produce mxk matrix C. The example kernel in PMPP assumes the matrices are square -- the examples here are more general as they do not make that assumption.

DETAILS:
'matrixMultiplication.cu' contains different example implementations of CUDA matrix multiplication kernels. Difference in implementations stems from how each launched GPU kernel thread is mapped to the output matrix C:
    case0: For 'threadToElementMMKernel', there is exactly one thread for every element in matrix C
    case1: For 'threadToRowMMKernel', there is exactly one thread for every row in matrix C
    case2: For 'threadToColumnMMKernel', there is exactly one thread for every column in matrix C

Block size is hardcoded to 16 threads per x- and y-dimension; the z-dimension for block size is hardcoded to 1 as the code assumes a maximum of 2 dimensions needed to represent any matrix.

User may specify desired implementation by setting 'kernel_choice' parameter in 'matrixMul()' accordingly.
