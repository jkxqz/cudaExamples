#include <stdio.h>

__global__
void matrixMulKernel(float* A, float* B, float* C, int height_A, int width_A, int width_B) {
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

void matrixMul(float* A_h, float* B_h, float* C_h, int height_A, int width_A, int width_B) {
    int size_A = sizeof(float)*height_A*width_A;
    int size_B = sizeof(float)*width_A*width_B;
    int size_C = sizeof(float)*height_A*width_B;

    // device pointers
    float* A_d;
    float* B_d;
    float* C_d;

    // allocating device on GPU
    cudaMalloc((void**) &A_d, size_A); 
    cudaMalloc((void**) &B_d, size_B); 
    cudaMalloc((void**) &C_d, size_C); 

    // copy array values from host to device
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width_B/16.0),ceil(height_A/16.0),1); // number of blocks in a grid
    dim3 dimBlock(16,16,1); // number of theads in a block
    matrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, height_A, width_A, width_B);

    // copy output array C from device to host
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    // deallocate memory on device
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    // A has dimension height_Axwidth_A
    // B has dimension height_Bxwidth_B -- height_B not used anywhere explicitly
    // C has dimension height_Axwidth_B
    
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

    //for (int i{0}; i < 4; ++i) {
    //    printf("%f, %f, %f \n", A[i], B[i], C[i]);
    //}

    matrixMul(A, B, C, height_A, width_A, width_B);


    for (int i{0}; i < 4; ++i) {
        printf("%f, %f, %f\n", A[i], B[i], C[i]);
    }

    return 0;
}

