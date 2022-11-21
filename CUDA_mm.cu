#include<cuda_runtime.h>
#define idx(x,y,M) (M*(x)+y)

__global__ void naive_mult(double *A, double *B, double *C, int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        double value = 0;
        for(int k = 0; k < size; k++){
            value += A[row*size + k] * B[k*size + col]; 
        }
        C[row*size + col] = value; 
    }
}

void mat_multiply(double* A, double* B, double* C, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.
    for(int i = 0; i < d11; ++i)
        for(int j = 0; j < d22; ++j)
            C[idx(i,j,d11)]=0;
    
    cudaEvent_t time_s, time_e;
    cudaEventCreate(&time_s);
    cudaEventCreate(&time_e);

    double *dA, *dB, *dC;
    cudaMalloc((void**)&dA,(d11*d12)*sizeof(double));
    cudaMalloc((void**)&dB,(d12*d22)*sizeof(double));
    cudaMalloc((void**)&dC,(d11*d22)*sizeof(double));

    cudaMemcpy(dA,A,(d11*d12)*(sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,(d12*d22)*(sizeof(double)),cudaMemcpyHostToDevice);
    cudaMemcpy(dC,C,(d11*d22)*(sizeof(double)),cudaMemcpyHostToDevice);

    dim3 tbp(8,8);
    dim3 numBlocks((d11/tbp.x<1)? 1:d11/tbp.x, (d11/tbp.y<1)? 1:d11/tbp.y);

    cudaEventRecord(time_s);
    naive_mult <<< numBlocks, tbp >>> (dA, dB, dC, d12);
    cudaEventRecord(time_e);
    cudaEventSynchronize(time_e);

    float time = 0;
    printf("Time the function took is %.4fs", cudaEventElapsedTime(&time, time_s, time_e));
    cudaMemcpy(C,dC,(d11*d22)*(sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return;
}
