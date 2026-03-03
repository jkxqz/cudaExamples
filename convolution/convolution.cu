// kernels for convolving a 2D matrix with a square filter
#include <stdio.h>

#define FILTER_RADIUS 1 // square convolution kernel of radius=1

// constant memory allocation for read-only copy of convolution kernel
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];


// TODO
__global__
void tiledHaloConvKernelT2In(float* A, float* B, int height, int width);

// TODO
__global__
void tiledHaloConvKernelT2Out(float* A, float* B, int height, int width);

__global__
void basicConvKernel(float* A, float* B, int height, int width) {
    // thread organization: one output thread per element in output matrix B

    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    // only allow thread to modify B if thread maps to actual element of B
    if (outCol < width && outRow < height) {
        float bValue = 0.0f; 
        int r = FILTER_RADIUS;
        for (int fRow=0; fRow<2*r+1; ++fRow) {
            for (int fCol=0; fCol<2*r+1; ++fCol) {
                int dR = fRow - r; 
                int dC = fCol - r;
                int inRow = outRow + dR;
                int inCol = outCol + dC;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    bValue += F_c[fRow][fCol]*A[inRow*width + inCol];
                }
            }
        }
        B[outRow*width + outCol] = bValue;
    }
    
}

void doConv(float* A_h, float* B_h, float* F_h, int height, int width) {
    
    // num bytes in A_h (same as in B_h)
    int size = width*height*sizeof(float);

    /* 
     allocate space for A_h and B_h on GPU global memory
     and store those memory addresses in A_d and B_d.
    */
    float* A_d;
    float* B_d;

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);

    // copy values of A_h into GPU memory just allocated
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    // copy values of F_h into constant GPU memory used for F_c
    cudaMemcpyToSymbol(F_c, F_h, sizeof(F_c));

    // arbitrarily setting blockSize to 16
    float blockSize = 16.0f;

    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(width/blockSize), ceil(height/blockSize), 1);

    // launch kernel to perform convolution
    basicConvKernel<<<dimGrid, dimBlock>>>(A_d, B_d, height, width);

    // convolution has been performed; now copy result from B_d back to host memory
    cudaMemcpy(B_h, B_d, size, cudaMemcpyDeviceToHost);
    
    // deallocate GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    
}


int main() {
    
    int width = 10;
    int height = 10;

    int fWidth = 2*FILTER_RADIUS+1;

    /* 
     allocating memory for input matrix A, output matrix B,
     and convolution kernel F.
    */
    float* A = (float *) malloc(width*height*sizeof(float));
    float* B = (float *) malloc(width*height*sizeof(float)); 
    float* F = (float *) malloc(sizeof(float)*fWidth*fWidth);


    if ( A==NULL || B==NULL || F==NULL ) {
        fprintf(stderr, "Memory allocation failure\n");
    }

    // Initializations
    for (int i=0; i<fWidth*fWidth; ++i) {
        F[i] = 2;
    }
    for (int i=0; i<width*height; ++i) {
        A[i] = 3;
    }

    //printf("%zu\n", sizeof(F_c)); // confirming F_c visibility and size

    doConv(A, B, F, height, width);

    //for (int i=0; i<width*height; ++i) {
    //    printf("%f\t%f\n", A[i], B[i]);
    //}

    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x)
            printf("%f\t", B[y*width + x]);
	printf("\n");
    }
    
    // memory deallocation
    free(A);
    free(B);
    free(F);

    return 0;
}

