SUMMARY:
Right multiply mxn matrix A by nxk matrix B to produce mxk matrix C. The example kernel in PMPP assumes the matrices are square -- the examples here do not make that assumption.

DETAILS:
Directory contains different examples of matrix multiplication. Difference is primarily how each gpu kernel thread is mapped to the output matrix C. For the 'oneColPerThread' case, each device thread is responsible for creating a its own column in matrix C. Similarly, 'oneRowPerThread' launches multiple gpu threads, but each thread computes all the elements for a single row in matrix C. 'oneElementPerThread' creates as many threads as there are elements in matrix C.
