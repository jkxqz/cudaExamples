#include <stdio.h>

// Tile width in x and y directions are set equal to
// the x and y dimensions of the block -- this assumes that
// the blocks are small enough to allow for tiles to fit in shared memory.
// TODO: This assumption should be relaxed more generally.
#define tileHeight 2
#define tileWidth 2

__global__
void tiledMMKernel(float* A, float* B, float* C, int height_A, int width_A, int width_B) {

    // shared memory variables on the device visible to all threads in a block
    __shared__ float Ads[tileHeight][tileWidth];
    __shared__ float Bds[tileHeight][tileWidth];

    // convenience variables
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify location in output matrix C that thread is computing a value for
    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;

    float Cvalue = 0;
    for (int phase=0; phase<width_A/tileWidth; ++phase) {
        // Collaborative loading of M and N tiles into shared memory
        Ads[ty][tx] = A[row*width_A + phase*tileWidth + tx];
        Bds[ty][tx] = B[(phase*tileHeight + ty)*width_B + col];
        __syncthreads();

        for (int k = 0; k < tileWidth; ++k) {
            Cvalue += Ads[ty][k]*Bds[k][tx];
        }

        __syncthreads();
    }

    C[row*width_A + col] = Cvalue;
}

void matrixMul(float* A_h, float* B_h, float* C_h, int height_A, int width_A, int width_B) {
    // these pointers will reference copies of A, B, C matrices
    float* A_d;
    float* B_d;
    float* C_d;

    int size_A = sizeof(float)*height_A*width_A;
    int size_B = sizeof(float)*width_A*width_B;
    int size_C = sizeof(float)*height_A*width_B;

    // Allocate memory on device for matrix copies
    cudaMalloc((void **) &A_d, size_A);
    cudaMalloc((void **) &B_d, size_B);
    cudaMalloc((void **) &C_d, size_C);

    // Make copies of A, B in device allocations
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    float blockSize = 2.0f;
    
    dim3 dimGrid(ceil(width_B/blockSize), ceil(height_A/blockSize), 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    tiledMMKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, height_A, width_A, width_B);

    // copy result on device memory back to host memory allocation of matrix C
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    A_d = B_d = C_d = NULL;

}

int main() {
    // matrix A has dimension height_A x width_A
    // matrix B has dimension height_B x width_B -- height_B must be equal to width_A
    // matrix C has dimension height_A x width_B
    int height_A = 4;
    int width_A = 4;
    int width_B = 4;

    float* A = (float *) malloc(sizeof(float)*height_A*width_A);
    float* B = (float *) malloc(sizeof(float)*width_A*width_B);
    float* C = (float *) malloc(sizeof(float)*height_A*width_B);

    if ((A==NULL) || (B==NULL) || (C==NULL)) {
        fprintf(stderr, "Host memory allocation failure\n");
        exit(1);
    }

    // Some initialization

    A[0] =  10; A[1] =  20; A[2] =  30; A[3] = 40; A[4] = 5; A[5] = 15; A[6] = 25; A[7] = 35; A[8] = 2; A[9] = 4; A[10] = 6; A[11] = 8; A[12] = 1; A[13] = 3; A[14] = 5; A[15] = 7;
    B[0] =  1; B[1] =  2; B[2] =  1; B[3] = 2; B[4] = 3; B[5] = 4; B[6] = 3; B[7] = 4; B[8] = 5; B[9] = 6; B[10] = 5; B[11] = 6; B[12] = 7; B[13] = 8; B[14] = 7; B[15] = 8;

    matrixMul(A, B, C, height_A, width_A, width_B);

    for (int i=0; i<16; ++i) {
        printf("%f, %f, %f\n", A[i], B[i], C[i]);
    }
    
    free(A); free(B); free(C);
    A = B = C = NULL;
    
    return 0;
}

