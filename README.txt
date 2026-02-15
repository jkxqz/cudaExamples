This repo contains selected exercises from the book "Programming Massively Parallel Processors" (PMPP).

The required resources for each selected problem (e.g., source code, data files) are all grouped in a shared subfolder for that problem.

All builds were successfully tested on a Google Colab instance using a single NVIDIA T4 GPU card (16GB DRAM) and CUDA Version 13.0 (Driver Version 580.82.07). Within this environment, the following command was run with the local NVIDIA CUDA compiler (nvcc) to create an executable:

nvcc <source_file> -o <out_file>

Any deviation from this general command format is mentioned where needed in the subfolder for a specific problem.
