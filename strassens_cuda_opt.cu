#include<cuda_runtime.h>
#include <stdlib.h>
#define idx(x,y,M) (M*(x)+(y))
using namespace std;

__global__ void gpu_add(double *A, double *B, double *C,
        int idx_Ar, int idx_Ac, 
        int idx_Br, int idx_Bc, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        C[idx(row,col,size)] = A[idx(row+idx_Ar,col+idx_Ac,size)] + B[idx(row+idx_Br,col+idx_Bc,size)]; 
    }
}

__global__ void gpu_sub(double *A, double *B, double *C, 
        int idx_Ar, int idx_Ac,
        int idx_Br, int idx_Bc, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        C[idx(row,col,size)] = A[idx(row+idx_Ar,col+idx_Ac,size)] - B[idx(row+idx_Br,col+idx_Bc,size)]; 
    }
}

__global__ void gpu_ext(double *A, double *B, 
        double *C_A00, double *C_B00,
        double *C_A21, double *C_B21, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        int idx_r = 0; int idx_c = 0;
        C_A00[idx(row,col,size)] = A[idx(row+idx_r,col+idx_c,size)]; 
        C_B00[idx(row,col,size)] = B[idx(row+idx_r,col+idx_c,size)]; 

        idx_r = size;
        idx_c = size;
        C_A21[idx(row,col,size)] = A[idx(row+idx_r,col+idx_c,size)]; 
        C_B21[idx(row,col,size)] = B[idx(row+idx_r,col+idx_c,size)]; 
    }
}

__global__ void mult_small(double *a, double *b, double *c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        double value = 0;
        for(int k = 0; k < size; k++){
            value += a[idx(row,col+k,size)] * b[idx(row+k,col,size)]; 
        }
        c[idx(row,col,size)] = value; 
    }
}

__global__ void gpu_synth(double *c11, double *c12, double *c21, double *c22, double* mult, int newSize) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int d11 = 2*newSize;

    if((i < newSize) && (j  < newSize)){
        mult[idx(i,j,d11)] = c11[idx(i,j,newSize)];
        mult[idx(i,(j + newSize),d11)] = c12[idx(i,j,newSize)];
        mult[idx((i + newSize),j,d11)] = c21[idx(i,j,newSize)];
        mult[idx((i + newSize),(j + newSize),d11)] = c22[idx(i,j,newSize)];
    }
}

void strassen_multiply(double* A, double* B, double* mult, int d11, cudaStream_t stream1, cudaStream_t stream2) {
    dim3 tbp(8,8);
    int newSize = d11/2;
    if(d11 <= 64) {
        dim3 numBlocks((d11/tbp.x<1)? 1:d11/tbp.x, (d11/tbp.y<1)? 1:d11/tbp.y);
        mult_small <<< numBlocks, tbp,0,stream1 >>> (A,B,mult,d11);
    } else {
        dim3 numBlocks((newSize/tbp.x<1)? 1:newSize/tbp.x, (newSize/tbp.y<1)? 1:newSize/tbp.y);

        double a11r = 0;
        double a11c = 0;
        double a12r = 0;
        double a12c = newSize;
        double a21r = newSize;
        double a21c = 0;
        double a22r = newSize;
        double a22c = newSize;

        double b11r = 0;
        double b11c = 0;
        double b12r = 0;
        double b12c = newSize;
        double b21r = newSize;
        double b21c = 0;
        double b22r = newSize;
        double b22c = newSize;

        double *dc11, *dc12, *dc21, *dc22;
        cudaMalloc((void**)&dc11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc12,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc21,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc22,(newSize*newSize)*sizeof(double));
        cudaMemset(dc11,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dc12,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dc21,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dc22,0,(newSize*newSize)*sizeof(double));

        double *dA11, *dB11, *dA22, *dB22;
        cudaMalloc((void**)&dA11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dA22,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB22,(newSize*newSize)*sizeof(double));
        cudaMemset(dA11,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dB11,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dA22,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dB22,0,(newSize*newSize)*sizeof(double));

        gpu_ext <<< numBlocks, tbp >>> (A, B, dA11, dB11, dA22, dB22, newSize);

        double *dS1, *dS2, *dS3, *dS4;
        cudaMalloc((void**)&dS1,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS2,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS3,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS4,(newSize*newSize)*sizeof(double));
        cudaMemset(dS1,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS2,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS3,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS4,0,(newSize*newSize)*sizeof(double));

        // s1 = a21 - a11
        // s2 = b11 + b12
        gpu_sub <<< numBlocks, tbp, 0, stream1>>> (A, A, dS1, a21r, a21c, a11r, a11c, newSize);
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (B, B, dS2, b11r, b11c, b12r, b12c, newSize);
        strassen_multiply(dS1, dS2, dc22, newSize,stream1,stream1);
        cudaMemset(dS1,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS2,0,(newSize*newSize)*sizeof(double));
        cudaStreamSynchronize(stream1);

        // s3 = a12 - a22
        // s4 = b21 + b22
        gpu_sub <<< numBlocks, tbp, 0, stream2 >>> (A, A, dS3, a12r, a12r, a22r, a22c, newSize);
        gpu_add <<< numBlocks, tbp, 0, stream2 >>> (B, B, dS4, b21r, b21c, b22r, b22c, newSize);
        strassen_multiply(dS3, dS4, dc11, newSize,stream2,stream2);
        cudaMemset(dS3,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS4,0,(newSize*newSize)*sizeof(double));
        cudaStreamSynchronize(stream2);

        // s1 = a21 + a22
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (A, A, dS1, a21r, a21c, a22r, a22c, newSize);
        strassen_multiply(dS1, dB11, dc21, newSize, stream1, stream1);
        gpu_sub <<< numBlocks, tbp, 0, stream1 >>> (dc22, dc21, dc22, 0, 0, 0, 0, newSize);
        cudaMemset(dS1,0,(newSize*newSize)*sizeof(double));
        cudaStreamSynchronize(stream1);

        // s3 = a11 + a12
        gpu_add <<< numBlocks, tbp, 0, stream2 >>> (A, A, dS3, a11r,a11c, a12r, a12c, newSize);
        strassen_multiply(dS3, dB22, dc12, newSize, stream2, stream2);
        gpu_sub <<< numBlocks, tbp, 0, stream2 >>> (dc11, dc12, dc11, 0, 0, 0, 0, newSize);
        cudaMemset(dS3,0,(newSize*newSize)*sizeof(double));
        cudaStreamSynchronize(stream2);


        // s1 = b12 - b22
        gpu_sub <<< numBlocks, tbp, 0, stream1 >>> (B, B, dS1, b12r, b12c, b22r, b22c, newSize);
        strassen_multiply(dA11, dS1, dS2, newSize, stream1, stream1);
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (dc12, dS2, dc12, 0, 0, 0, 0, newSize);
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (dc22, dS2, dc12, 0, 0, 0, 0, newSize);
        cudaMemset(dS1,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS2,0,(newSize*newSize)*sizeof(double));
        cudaStreamSynchronize(stream1);

        // s3 = b21 - b11
        gpu_sub <<< numBlocks, tbp, 0, stream2 >>> (B, B, dS3, b21r, b21c, b11r, b11c, newSize);
        strassen_multiply(dA22, dS3, dS4, newSize, stream2, stream2);
        gpu_add <<< numBlocks, tbp, 0, stream2 >>> (dc11, dS4, dc11, 0, 0, 0, 0, newSize);
        gpu_add <<< numBlocks, tbp, 0, stream2 >>> (dc21, dS4, dc21, 0, 0, 0, 0, newSize);
        cudaMemset(dS3,0,(newSize*newSize)*sizeof(double));
        cudaMemset(dS4,0,(newSize*newSize)*sizeof(double));
        cudaStreamSynchronize(stream2);
        
        // s1 = a11 + a22
        // s2 = b11 + b22
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (A, A, dS1, a11r, a11c, a22r, a22c, newSize);
        gpu_add <<< numBlocks, tbp, 0, stream2 >>> (B, B, dS2, b11r, b11c, b22r, b22c, newSize);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        strassen_multiply(dS1, dS2, dS3, newSize, stream1, stream1);
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (dc11, dS3, dc11, 0, 0, 0, 0, newSize);
        gpu_add <<< numBlocks, tbp, 0, stream1 >>> (dc22, dS3, dc22, 0, 0, 0, 0, newSize);

        gpu_synth <<< numBlocks, tbp,0,stream1 >>> (dc11,dc12,dc21,dc22,mult,newSize);

        cudaFree(dS1);
        cudaFree(dS2);
        cudaFree(dS3);
        cudaFree(dS4);
        cudaFree(dA11);
        cudaFree(dA22);
        cudaFree(dB11);
        cudaFree(dB22);

        cudaFree(dc11);
        cudaFree(dc12);
        cudaFree(dc21);
        cudaFree(dc22);
    }
}

void strassen_root(double* a, double* b, double* mult, int d11) {
    double *da, *db, *dc;
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMalloc((void**)&da,(d11*d11)*sizeof(double));
    cudaMemcpy(da,a,(d11*d11)*(sizeof(double)),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&db,(d11*d11)*sizeof(double));
    cudaMemcpy(db,b,(d11*d11)*(sizeof(double)),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dc,(d11*d11)*sizeof(double));
    cudaMemset(dc,0,(d11*d11)*sizeof(double));

    strassen_multiply(da,db,dc,d11,stream1,stream2);
    cudaMemcpy(mult,dc,(d11*d11)*(sizeof(double)),cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

void mat_multiply(double* a, double* b, double* mult, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.

    cudaEvent_t time_s, time_e;
    cudaEventCreate(&time_s);
    cudaEventCreate(&time_e);

    cudaEventRecord(time_s);
    strassen_root(a,b,mult,d11);
    cudaEventRecord(time_e);
    cudaEventSynchronize(time_e);

    float time = 0;
    cudaEventElapsedTime(&time, time_s, time_e);

    printf("Time the function took is %.5fns", time);
    return;
}
