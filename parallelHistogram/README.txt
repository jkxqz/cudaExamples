SUMMARY:
Create a histogram for user-provided input string, where each histogram cell corresponds to the number of occurrences of a character in the input string. The number of characters tracked is equal to the size of the set of lowercase English letters [a-z] (26).

DETAILS:
'parallelHistogram.cu' shows a kernel with a 1-dimensional grid and 1-dimensional block. The key part of the implementation is the use of a CUDA intrinsic 'atomicAdd', whereby the histogram cell update is effectively compiled to an atomic operation. Removing this update and leaving the program otherwise the same would introduce a race condition.

HISTOGRAM_SIZE set as symbolic constant at top of file (26).

