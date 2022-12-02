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

__global__ void gpu_add(double *A, double *B, double *C, int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        C[row*size+col] = A[row*size + col] + B[row*size + col]; 
    }
}

__global__ void gpu_sub(double *A, double *B, double *C, int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        C[row*size+col] = A[row*size + col] - B[row*size + col]; 
    }
}

void mult_small(double* a, double* b, double* c, int size){
    for(int i = 0; i < size; ++i)
        for(int j = 0; j < size; ++j)
            for(int k = 0; k < size; ++k)
                c[idx(i,j,size)] += a[idx(i,k,size)] * b[idx(k,j,size)];
}

void strassen_multiply(double* a, double* b, double* mult, int d11, int d12, int d22,int og_size) {
    for(int i = 0; i < d11; ++i)
        for(int j = 0; j < d22; ++j)
            mult[idx(i,j,d11)]=0;

    if(d11 <= 32) {
        mult_small(a,b,mult,d11);
        return; 
    } else {
        int newSize = d11/2;
        double a11[newSize*newSize];
        double a12[newSize*newSize];
        double a21[newSize*newSize];
        double a22[newSize*newSize];
        double b11[newSize*newSize];
        double b12[newSize*newSize];
        double b21[newSize*newSize];
        double b22[newSize*newSize];
        double c11[newSize*newSize];
        double c12[newSize*newSize];
        double c21[newSize*newSize];
        double c22[newSize*newSize];
        double s1[newSize*newSize];
        double s2[newSize*newSize];
        double s3[newSize*newSize];
        double s4[newSize*newSize];
        double s5[newSize*newSize];
        double s6[newSize*newSize];
        double s7[newSize*newSize];
        double s8[newSize*newSize];
        double s9[newSize*newSize];
        double s10[newSize*newSize];
        double p1[newSize*newSize];
        double p2[newSize*newSize];
        double p3[newSize*newSize];
        double p4[newSize*newSize];
        double p5[newSize*newSize];
        double p6[newSize*newSize];
        double p7[newSize*newSize];
        double tempA[newSize*newSize];
        double tempB[newSize*newSize];

        for (int i = 0; i < newSize; i++) {
            for (int j = 0; j < newSize; j++) {
                a11[idx(i,j,newSize)] = a[idx(i,j,d11)];
                a12[idx(i,j,newSize)] = a[idx(i,(j + newSize),d11)];
                a21[idx(i,j,newSize)] = a[idx((i + newSize),j,d11)];    
                a22[idx(i,j,newSize)] = a[idx((i + newSize),(j + newSize),d11)];

                b11[idx(i,j,newSize)] = b[idx(i,j,d11)];
                b12[idx(i,j,newSize)] = b[idx(i,(j + newSize),d11)];
                b21[idx(i,j,newSize)] = b[idx((i + newSize),j,d11)];
                b22[idx(i,j,newSize)] = b[idx((i + newSize),(j + newSize),d11)];
            }
        }

        double *dA11, *dA12, *dA21, *dA22, *dB11, *dB12, *dB21, *dB22, *dS1, *dS2, *dS3, *dS4, *dS5, *dS6, *dS7, *dS8, *dS9, *dS10;
        cudaMalloc((void**)&dA11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dA12,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dA21,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dA22,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB12,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB21,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dB22,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS1,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS2,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS3,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS4,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS5,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS6,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS7,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS8,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS9,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dS10,(newSize*newSize)*sizeof(double));

        cudaMemcpy(dA11,a11,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dA12,a12,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dA21,a21,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dA22,a22,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dB11,b11,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dB12,b12,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dB21,b21,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dB22,b22,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);

        cudaMemcpy(dS1,s1,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS2,s2,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS3,s3,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS4,s4,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS5,s5,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS6,s6,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS7,s7,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS8,s8,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS9,s9,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dS10,s10,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);

        dim3 tbp(8,8);
        dim3 numBlocks((newSize/tbp.x<1)? 1:newSize/tbp.x, (newSize/tbp.y<1)? 1:newSize/tbp.y);

        // s1 = b12 - b22
        gpu_sub <<< numBlocks, tbp >>> (dB12, dB22, dS1, newSize);
        
        // s2 = a11 + a12
        gpu_add <<< numBlocks, tbp >>> (dA11, dA12, dS2, newSize);

        // s3 = a21 + a22
        gpu_add <<< numBlocks, tbp >>> (dA21, dA22, dS3, newSize);

        // s4 = b21 - b11
        gpu_sub <<< numBlocks, tbp >>> (dB21, dB11, dS4, newSize);
        
        // s5 = a11 + a22
        gpu_add <<< numBlocks, tbp >>> (dA11, dA22, dS5, newSize);
        
        // s6 = b11 + b22
        gpu_add <<< numBlocks, tbp >>> (dB11, dB22, dS6, newSize);

        // s7 = a12 - a22
        gpu_sub <<< numBlocks, tbp >>> (dA12, dA22, dS7, newSize);

        // s8 = b21 + b22
        gpu_add <<< numBlocks, tbp >>> (dB21, dB22, dS8, newSize);

        // s9 = a21 - a11
        gpu_sub <<< numBlocks, tbp >>> (dA21, dA11, dS9, newSize);

        // s10 = b11 + b12
        gpu_add <<< numBlocks, tbp >>> (dB11, dB12, dS10, newSize);

        cudaDeviceSynchronize();

        cudaMemcpy(s1,dS1,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s2,dS2,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s3,dS3,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s4,dS4,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s5,dS5,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s6,dS6,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s7,dS7,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s8,dS8,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s9,dS9,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(s10,dS10,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);

        strassen_multiply(s7, s8, p1, newSize, newSize, newSize, newSize);

        strassen_multiply(s5, s6, p2, newSize, newSize, newSize, newSize);

        strassen_multiply(s9, s10, p3, newSize, newSize, newSize, newSize);

        strassen_multiply(s2, b22, p4, newSize, newSize, newSize, newSize);

        strassen_multiply(a11, s1, p5, newSize, newSize, newSize, newSize);

        strassen_multiply(a22, s4, p6, newSize, newSize, newSize, newSize);

        strassen_multiply(s3, b11, p7, newSize, newSize, newSize, newSize);

        cudaFree(dA11);
        cudaFree(dA12);
        cudaFree(dA21);
        cudaFree(dA22);
        cudaFree(dB11);
        cudaFree(dB12);
        cudaFree(dB21);
        cudaFree(dB22);

        cudaFree(dS1);
        cudaFree(dS2);
        cudaFree(dS3);
        cudaFree(dS4);
        cudaFree(dS5);
        cudaFree(dS6);
        cudaFree(dS7);
        cudaFree(dS8);
        cudaFree(dS9);
        cudaFree(dS10);

        double *dp1, *dp2, *dp3, *dp4, *dp5, *dp6, *dp7, *dtempA, *dtempB, *dc11, *dc12, *dc21, *dc22;
        cudaMalloc((void**)&dp1,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dp2,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dp3,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dp4,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dp5,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dp6,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dp7,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dtempA,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dtempB,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc11,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc12,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc21,(newSize*newSize)*sizeof(double));
        cudaMalloc((void**)&dc22,(newSize*newSize)*sizeof(double));

        cudaMemcpy(dp1,p1,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dp2,p2,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dp3,p3,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dp4,p4,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dp5,p5,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dp6,p6,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dp7,p7,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dtempA,tempA,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dtempB,tempB,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dc11,c11,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dc12,c12,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dc21,c21,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);
        cudaMemcpy(dc22,c22,(newSize*newSize)*(sizeof(double)),cudaMemcpyHostToDevice);


        // c11 = p1 + p2 - p4 + p6
        gpu_add <<< numBlocks, tbp >>> (dp1, dp2, dtempA, newSize); // p1 + p2
        gpu_add <<< numBlocks, tbp >>> (dtempA, dp6, dtempB, newSize); // (p1 + p2) + p6
        gpu_sub <<< numBlocks, tbp >>> (dtempB, dp4, dc11, newSize); // (p5 + p4 + p6) - p2

        // c12 = p4 + p5
        gpu_add <<< numBlocks, tbp >>> (dp4, dp5, dc12, newSize);

        // c21 = p6 + p7
        gpu_add <<< numBlocks, tbp >>> (dp6, dp7, dc21, newSize);

        // c22 = p2 - p3 + p5 - p7
        gpu_add <<< numBlocks, tbp >>> (dp2, dp3, dtempA, newSize); // p2 - p3
        gpu_sub <<< numBlocks, tbp >>> (dtempA, dp7, dtempB, newSize); // (p2 - p3) - p7
        gpu_add <<< numBlocks, tbp >>> (dtempB, dp5, dc22, newSize); // (p2 - p3 - p7) + p5

        cudaMemcpy(c11,dc11,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(c12,dc12,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(c21,dc21,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);
        cudaMemcpy(c22,dc22,(newSize*newSize)*(sizeof(double)),cudaMemcpyDeviceToHost);

        cudaFree(dp1);
        cudaFree(dp2);
        cudaFree(dp3);
        cudaFree(dp4);
        cudaFree(dp5);
        cudaFree(dp6);
        cudaFree(dp7);
        cudaFree(dtempA);
        cudaFree(dtempB);
        cudaFree(dc11);
        cudaFree(dc12);
        cudaFree(dc21);
        cudaFree(dc22);

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
void mat_multiply(double* a, double* b, double* mult, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.

    cudaEvent_t time_s, time_e;
    cudaEventCreate(&time_s);
    cudaEventCreate(&time_e);

    cudaEventRecord(time_s);
    strassen_multiply(a,b,mult,d11,d12,d22,d11);
    cudaEventRecord(time_e);
    cudaEventSynchronize(time_e);

    float time = 0;
    cudaEventElapsedTime(&time, time_s, time_e);

    printf("Time the function took is %.5fns", time);
    return;
}