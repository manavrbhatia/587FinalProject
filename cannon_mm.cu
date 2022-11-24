#include<cuda_runtime.h>

#define idx(x,y,M) (M*(x)+(y))

__global__ void cannon_mult(double *A, double *B, double *C, const int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double shared_row[blockDim.y];
    __shared__ double shared_col[blockDim.x];

    if((row < size) && (col  < size)){
        int k = (row + col) % size;
        int el_a = A[idx(row,k,size)];
        int el_b = B[idx(k,j,size)];
    
        for(int i = 0; i < size; i++){
            C[row*size + col] = C[row*size + col] + el_a*el_b; 
            shared_row[threadIdx.y] = el_a;
            shared_col[threadIdx.x] = el_b; 
            __syncthreads();
            el_a = shared_row[(threadIdx.x+1) % size];
            el_b = shared_col[(threadIdx.y+1) % size];
        }
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
    cannon_mult <<< numBlocks, tbp >>> (dA, dB, dC, d12);
    cudaEventRecord(time_e);
    cudaEventSynchronize(time_e);

    float time = 0;
    cudaEventElapsedTime(&time, time_s, time_e);
    printf("Time the function took is %.5fns", time);
    cudaMemcpy(C,dC,(d11*d22)*(sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return;
}
