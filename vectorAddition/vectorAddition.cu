#include <stdio.h>

// Compute vector sum C = A + B
// Each thread performs one pair-wise addition
__global__ // cuda-c-specific keyword
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    
    float* A_d;
    float* B_d;
    float* C_d;

    // Allocating memory on device 
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    // copy memory from host to device
    // only need to transfer A_h and B_h
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // kernel invocation code
    // launch ceil(n/256) blocks of 256 threads each
    vecAddKernel<<<ceil((int) n/256.0), 256>>>(A_d, B_d, C_d, n);
  
    // copy memory from device to host
    // only want the C_d array
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // free the memory on the device when done
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // allocating memory for arrays on host
    int n = 1'000'000;
    float* A = (float *) malloc(n*sizeof(float));
    float* B = (float *) malloc(n*sizeof(float));
    float* C = (float *) malloc(n*sizeof(float));

    // pointer provided by malloc will be 'NULL' if malloc failed
	// to allocate memory succesfully. In case of failure, notify
    // user and exit program.
    if ((A == NULL) || (B==NULL) || (C==NULL)) {
        fprintf(stderr, "Host memory allocation failure\n");
        exit(1);
    }

    // initialize A and B with 'random' data
    for (int i=0; i<n; ++i) {
        A[i] = n*2 - i*n + n/32;
        B[i] = -n*2 - i*n + n/32 + i*i;
    }

    // Basic check of addition by looking at first element
    printf("Before: %f %f %f\n", A[0], B[0], C[0]);

    vecAdd(A, B, C, n);

    printf("After: %f %f %f\n", A[0], B[0], C[0]); 
    

    return 0;
}

