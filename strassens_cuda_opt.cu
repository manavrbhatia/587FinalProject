#include<cuda_runtime.h>
#include <stdlib.h>
#define idx(x,y,M) (M*(x)+(y))
using namespace std;

/* Performs strassens matrix multiplication using given matrices and their dimension.
 *
 * Returns a dynamically allocated flattened array for matrix containing result
 *
 * Arguments:
 * a,b: Matrices you want to multiple;
 * mult: Matrix you want to return results into;
 * int d11: Number of rows in a;
 * int d12: Number of columns in a, note that d12=d21 so dont need both;
 * int d22: Number of columns in b;
 * a * b = mult -> (d11,d12) * (d12,d22) = (d11,d22)
 */

__global__ void gpu_add(double *A, double *B, double *C,
        int idx_Ar, int idx_Br, 
        int idx_Ac, int idx_Bc, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        //C[idx(row+idx_Cr,col+idx_Cc,size)] = A[idx(row+idx_Ar,col+idx_Ac,size)] + B[idx(row+idx_Br,col+idx_Bc,size)]; 
        C[idx(row,col,size)] = A[idx(row+idx_Ar,col+idx_Ac,size)] + B[idx(row+idx_Br,col+idx_Bc,size)]; 
    }
}

__global__ void gpu_sub(double *A, double *B, double *C, 
        int idx_Ar, int idx_Br,
        int idx_Ac, int idx_Bc, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        //C[idx(row+idx_Cr,col+idx_Cc,size)] = A[idx(row+idx_Ar,col+idx_Ac,size)] - B[idx(row+idx_Br,col+idx_Bc,size)]; 
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

        int idx_r = size; int idx_c = 0;
        C_A21[idx(row,col,size)] = A[idx(row+idx_r,col+idx_c,size)]; 
        C_B21[idx(row,col,size)] = B[idx(row+idx_r,col+idx_c,size)]; 
    }
}

__global__ void mult_small(double *a, double *b, double *c,
        int idx_Ar, int idx_Br, 
        int idx_Ac, int idx_Bc, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        double value = 0;
        for(int k = 0; k < size; k++){
            value += a[idx(row_idx_Ar,idx_Ac+k,size)] * b[idx(idx_Br+k,idx_Bc+col,size)]; 
        }
        c[idx(row,col,size)] = value; 
    }
}

__global__ void gpu_synth(double *c11, double *c12, double *c21, double *c22, int mult, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        for (int i = 0; i < newSize ; i++) {
            for (int j = 0 ; j < newSize ; j++) {
                mult[idx(i,j,d11)] = c11[idx(i,j,newSize)];
                mult[idx(i,(j + newSize),d11)] = c12[idx(i,j,newSize)];
                mult[idx((i + newSize),j,d11)] = c21[idx(i,j,newSize)];
                mult[idx((i + newSize),(j + newSize),d11)] = c22[idx(i,j,newSize)];
            }
        }
    }
}

void strassen_multiply(double* a, double* b, double* mult, int d11) {
    if(d11 <= 64) {
        dim3 tbp(8,8);
        dim3 numBlocks((d11/tbp.x<1)? 1:d11/tbp.x, (d11/tbp.y<1)? 1:d11/tbp.y);

        mult_small <<< numBlocks, tbp >>> (a,b,mult,0,0,0,0,d11);
        return; 
    } else {
        int newSize = d11/2;
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

        double *dA11, *dB11, *dA22, *dB22;
        cudaMalloc((void**)&dB11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dA11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dA22,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB22,(newSize*newSize)*sizeof(double));
        gpu_ext <<< numBlocks, tbp >>> (A, B, dA11, dB11, dA22, dB22, newSize);

        double *dS1, *dS2, *dS3, *dS4;
        cudaMalloc((void**)&dS1,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS2,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS3,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS4,(newSize*newSize)*sizeof(double));

        dim3 tbp(8,8);
        dim3 numBlocks((newSize/tbp.x<1)? 1:newSize/tbp.x, (newSize/tbp.y<1)? 1:newSize/tbp.y);

        // C11=s9*s10
        // s9 = a21 - a11
        // s10 = b11 + b12
        gpu_sub <<< numBlocks, tbp >>> (A, A, dS1, a21r, a21c, a11r, a11c, newSize);
        gpu_add <<< numBlocks, tbp >>> (B, B, dS2, b11r, b11c, b12r, b12c, newSize);
        strassen_multiply(dS1, dS2, dc22, newSize, newSize, newSize, newSize);

        // C22=s7*s8
        // s7 = a12 - a22
        // s8 = b21 + b22
        gpu_sub <<< numBlocks, tbp >>> (A, A, dS3, a12r, a12r, a22r, a22c, newSize);
        gpu_add <<< numBlocks, tbp >>> (B, B, dS4, b21r, b21c, b22r, b22c, newSize);
        strassen_multiply(dS3, dS4, dc11, newSize, newSize, newSize, newSize);

        // s3 = a21 + a22
        gpu_add <<< numBlocks, tbp >>> (A, A, dS1, a21r, a21c, a22r, a22c, newSize);
        strassen_multiply(dS1, dB11, dc21, newSize);
        gpu_sub <<< numBlocks, tbp >>> (dc22, dc21, dc22, 0, 0, 0, 0, newSize);

        // s2 = a11 + a12
        gpu_add <<< numBlocks, tbp >>> (A, A, dS3, da11r, da11c, da12r, da12c, newSize);
        strassen_multiply(dS3, dB22, dc12, newSize, newSize, newSize, newSize);
        gpu_sub <<< numBlocks, tbp >>> (dc11, dc12, dc11, 0, 0, 0, 0, newSize);



        // s1 = b12 - b22
        gpu_sub <<< numBlocks, tbp >>> (B, B, dS1, db12r, db12c, db22r, db22c, newSize);
        strassen_multiply(dA11, dS1, dS2, newSize, newSize, newSize, newSize);
        gpu_add <<< numBlocks, tbp >>> (dc12, dS2, dc12, 0, 0, 0, 0, newSize);
        gpu_add <<< numBlocks, tbp >>> (dc22, dS2, dc12, 0, 0, 0, 0, newSize);

        // s4 = b21 - b11
        gpu_sub <<< numBlocks, tbp >>> (B, B, dS3, b21r, b21c, b11r, b11c, newSize);
        strassen_multiply(dA22, s4, dS4, newSize, newSize, newSize, newSize);
        gpu_add <<< numBlocks, tbp >>> (dc11, dS4, dc11, 0, 0, 0, 0, newSize);
        gpu_add <<< numBlocks, tbp >>> (dc21, dS4, dc21, 0, 0, 0, 0, newSize);
        
        // s5 = a11 + a22
        // s6 = b11 + b22
        gpu_add <<< numBlocks, tbp >>> (A, A, dS1, a11r, a11c, a22r, a22c, newSize);
        gpu_add <<< numBlocks, tbp >>> (B, B, dS2, b11r, b11c, b22r, b22c, newSize);
        strassen_multiply(dS1, dS2, dS3, newSize, newSize, newSize, newSize);
        gpu_add <<< numBlocks, tbp >>> (dc11, dS3, dc11, 0, 0, 0, 0, newSize);
        gpu_add <<< numBlocks, tbp >>> (dc22, dS3, dc22, 0, 0, 0, 0, newSize);

        cudaDeviceSynchronize();

        cudaFree(dS1);
        cudaFree(dS2);
        cudaFree(dS3);
        cudaFree(dS4);

        cudaMemcpy(c11,dc11,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(c12,dc12,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(c21,dc21,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(c22,dc22,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);

        cudaFree(dc11);
        cudaFree(dc12);
        cudaFree(dc21);
        cudaFree(dc22);

        gpu_synth <<< numBlocks, tbp >>> (dc11,dc12,dc21,dc22,mult);
    }
}

void strassen_root(double* a, double* b, double* mult, int d11) {
    double * da, db, dc;
    cudaMalloc((void**)&da,(d11*d11)*sizeof(double));
    cudaMalloc((void**)&db,(d11*d11)*sizeof(double));
    cudaMalloc((void**)&dc,(d11*d11)*sizeof(double));
    strassen_mult(da,db,dc,d11);
    cudaMemcpy(mult,dc,(d11*d11)*(sizeof(double)),cudaMemcpyDeviceToHost);

}

void mat_multiply(double* a, double* b, double* mult, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.

    cudaEvent_t time_s, time_e;
    cudaEventCreate(&time_s);
    cudaEventCreate(&time_e);

    cudaEventRecord(time_s);
    strassen_root(a,b,mult,d11,d12,d22,d11);
    cudaEventRecord(time_e);
    cudaEventSynchronize(time_e);

    float time = 0;
    cudaEventElapsedTime(&time, time_s, time_e);

    printf("Time the function took is %.5fns", time);
    return;
}
