// Created by jkxqz [Feb 2026]
// matrixMultiplication.cu -- naive nonoptimized implementations of CUDA matrix multiplication kernels

#include <stdio.h>

__global__
void threadToElementMMKernel(float* A, float* B, float* C, int height_A, int width_A, int width_B) {
    // 1 thread for every element in output matrix C
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if (row < height_A && col < width_B) {
        float accumulated_sum = 0;
        for (int k=0; k<width_A; ++k) {
            accumulated_sum += A[row*width_A+k] * B[k*width_B + col];
        }
        C[row*width_B + col] = accumulated_sum;
    }
}

__global__
void threadtoRowMMKernel(float* A, float* B, float* C, int height_A, int width_A, int width_B) {
    // 1 thread for every row in output matrix C
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    if (row < height_A) {
        float accumulated_sum;
        for (int col=0; col<width_B; ++col) {
            accumulated_sum = 0;
            for (int k=0; k<width_A; ++k) {
                accumulated_sum += A[row*width_A+k] * B[k*width_B + col];
            }
            C[row*width_B + col] = accumulated_sum;
        }
    }
}

__global__
void threadtoColumnMMKernel(float* A, float* B, float* C, int height_A, int width_A, int width_B) {
    // 1 thread for every col in output matrix C
    //int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    if (col < width_B) {
        float accumulated_sum;
        for (int row=0; row<height_A; ++row) {
            accumulated_sum = 0;
            for (int k=0; k<width_A; ++k) {
                accumulated_sum += A[row*width_A+k] * B[k*width_B + col];
            }
            C[row*width_B + col] = accumulated_sum;
        }
    }
}

void matrixMul(float* A_h, float* B_h, float* C_h, int height_A, int width_A, int width_B, int kernel_choice=0) {
    // space (bytes) in device global memory needed to hold copies of A, B, and C matrices
    int size_A = sizeof(float)*height_A*width_A;
    int size_B = sizeof(float)*width_A*width_B;
    int size_C = sizeof(float)*height_A*width_B;

    // device global memory pointers
    float* A_d;
    float* B_d;
    float* C_d;

    // allocating memory on device
    cudaMalloc((void**) &A_d, size_A);
    cudaMalloc((void**) &B_d, size_B);
    cudaMalloc((void**) &C_d, size_C);

    // copy array values from host to device
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    // desired number of threads per dimension in a block 
    // (x and y only -- z dimension always 1 for these kernels)
    float blockSize = 16.0f;

    // execute 1 of the 3 kernel options
    switch(kernel_choice) {
    case 0: {// one thread per element of C matrix (default)
        dim3 dimGrid(ceil(width_B/blockSize),ceil(height_A/blockSize),1); // number of blocks in a grid
        dim3 dimBlock(blockSize,blockSize,1); // number of theads in a block
        threadToElementMMKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, height_A, width_A, width_B);
        break;
    }

    case 1: {// one thread per row of C matrix
        dim3 dimGrid(1,ceil(height_A/blockSize),1); // number of blocks in a grid
        dim3 dimBlock(1,blockSize,1); // number of theads in a block
        threadtoRowMMKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, height_A, width_A, width_B);
        break;
    }
    case 2: {// one thread per column of C matrix
        dim3 dimGrid(ceil(width_B/blockSize), 1, 1);
        dim3 dimBlock(blockSize, 1, 1);
        threadtoColumnMMKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, height_A, width_A, width_B);
	break;
    }
    default:
        fprintf(stderr, "ERROR\n");
        exit(1);
    }

    // copy output matrix C from device to host
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    // deallocate memory on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // matrix A has dimension height_A x width_A
    // matrix B has dimension height_B x width_B -- height_B not used anywhere explicitly
    // matrix C has dimension height_A x width_B

    int height_A = 2;
    int width_A = 2;
    int width_B = 2;

    float* A = (float *) malloc(sizeof(float)*height_A*width_A);
    float* B = (float *) malloc(sizeof(float)*width_A*width_B);
    float* C = (float *) malloc(sizeof(float)*height_A*width_B);

    if ((A==NULL) || (B==NULL) || (C==NULL)) {
        fprintf(stderr, "Host memory allocation failure\n");
        exit(1);
    }

    // Some initialization
    A[0] = 3; A[1] = 2; A[2] = 1; A[3] = -5;
    B[0] = 2; B[1] = 0; B[2] = -3; B[3] = 3;

    matrixMul(A, B, C, height_A, width_A, width_B, 1);

    for (int i{0}; i < 4; ++i) {
        printf("%f, %f, %f\n", A[i], B[i], C[i]);
    }

    return 0;
}



