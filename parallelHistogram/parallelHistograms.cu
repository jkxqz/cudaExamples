/*
 parallelHistograms.cu:

 provided some input string, use a GPU kernel to update a character-valued histogram
*/
#include <stdio.h>
#include <string.h> // for strlen()

#define HISTOGRAM_LENGTH 26 // 1 cell for each lowercase English letter



__global__
void populateHistogram(int* hist, int strLength, const char* inputString) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < strLength) {
        int alphabet_position = inputString[index] - 'a';
        if (alphabet_position >=0  && alphabet_position < HISTOGRAM_LENGTH) {
            atomicAdd(&hist[alphabet_position], 1); // atmoicAdd is CUDA intrinsic for atomized increments
        }
    }
}

void doHist(int* hist_h, int strLength, const char* inputString_h) {
    int*  hist_d;
    int   histSize = sizeof(int)*HISTOGRAM_LENGTH;
    char* inputString_d;
    int   stringSize = sizeof(char) * strLength;

    cudaMalloc((void**) &hist_d, histSize);
    cudaMalloc((void**) &inputString_d, stringSize);

    cudaMemcpy(hist_d, hist_h, histSize, cudaMemcpyHostToDevice);
    cudaMemcpy(inputString_d, inputString_h, stringSize, cudaMemcpyHostToDevice); 

    float blockSize = 32;

    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(ceil(strLength/blockSize), 1, 1);

    populateHistogram<<<dimGrid, dimBlock>>>(hist_d, strLength, inputString_d);

    cudaMemcpy(hist_h, hist_d, histSize, cudaMemcpyDeviceToHost);

    cudaFree(hist_d);
    cudaFree(inputString_d);
    
}

int main(int argc, char** argv) {
    const char* inputString;

    if (argc == 2) {
        inputString = argv[1];
    } else {
	inputString = "default string";
    }

    int strLength = (int) strlen(inputString);
    int hist[HISTOGRAM_LENGTH]; 

    for (int i=0; i<HISTOGRAM_LENGTH; ++i) {
        hist[i] = 0;
    }

    doHist(hist, strLength, inputString);

    printf("Counts of lowercase English letters in '%s':\n\n", inputString);
    for (int i=0; i<HISTOGRAM_LENGTH; ++i) {
        printf("%d ", hist[i]);
    }
    printf("\n");

    return 0;
}

