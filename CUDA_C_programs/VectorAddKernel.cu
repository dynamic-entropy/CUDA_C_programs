#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ERR_CHK(call) { gpuAssert((call), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t err, const char* file, int line, bool abort = true)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

__global__ void vecAddKernel(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


int main() {

    int* d_A, * d_B, * d_C;
    int* h_A, * h_B, * h_C;
    const int n = 1024;

    h_A = (int*)malloc(n * sizeof(int));
    h_B = (int*)malloc(n * sizeof(int));
    h_C = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        h_A[i] = rand();
        h_B[i] = rand();
        h_C[i] = 0;
    }

    ERR_CHK(cudaMalloc((void**)&d_A, n * sizeof(int)));
    ERR_CHK(cudaMalloc((void**)&d_B, n * sizeof(int)));
    ERR_CHK(cudaMalloc((void**)&d_C, n * sizeof(int)));

    ERR_CHK(cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice));
    ERR_CHK(cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice));


    dim3 gridSize(ceil(n / 256), 1, 1);
    dim3 blockSize(256, 1, 1);
    vecAddKernel <<< gridSize, blockSize >>> (d_A, d_B, d_C, n);
    cudaError_t err = cudaGetLastError();
    ERR_CHK(err);

    ERR_CHK(cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost));



    //verifying our solution
    for (int i = 0; i < n; i++) {
        if (h_A[i] + h_B[i] != h_C[i]) {
            printf("Incorrect addition");
            printf("%d + %d = %d for i = %d\n", h_A[i], h_B[i], h_C[i], i);
        }

    }
    printf("SUCCESS!!!!!!!!!!!");
    return 0;
}
