#include <time.h>
#define idx(x,y,M) (M*(x)+y)
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

void add(double* a, double* b, double* c, int size){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[idx(i,j,size)] = a[idx(i,j,size)] + b[idx(i,j,size)];
        }
    }
}

void sub(double* a, double* b, double* c, int size){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[idx(i,j,size)] = a[idx(i,j,size)] - b[idx(i,j,size)];
        }
    }
}

void mat_multiply(double* a, double* b, double* mult, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.
    for(int i = 0; i < d11; ++i)
        for(int j = 0; j < d22; ++j)
            mult[idx(i,j,d11)]=0;

    if(d11 == 1){
        mult[idx(0,0,d11)] = a[idx(0,0,d11)]*b[idx(0,0,d11)];
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
                a11[idx(i,j,d11)] = a[idx(i,j,d11)];
                a12[idx(i,j,d11)] = a[idx(i,(j + newSize),d11)];
                a21[idx(i,j,d11)] = a[idx((i + newSize),j,d11)];    
                a22[idx(i,j,d11)] = a[idx((i + newSize),(j + newSize),d11)];

                b11[idx(i,j,d11)] = b[idx(i,j,d11)];
                b12[idx(i,j,d11)] = b[idx(i,(j + newSize),d11)];
                b21[idx(i,j,d11)] = b[idx((i + newSize),j,d11)];
                b22[idx(i,j,d11)] = b[idx((i + newSize),(j + newSize),d11)];
            }
        }

        // s1 = b12 - b22
        sub(b12, b22, s1, newSize);
        
        // s2 = a11 + a12
        add(a11, a12, s2, newSize);

        // s3 = a21 + a22
        add(a21, a22, s3, newSize);

        // s4 = b21 - b11
        sub(b21, b11, s4, newSize);
        
        // s5 = a11 + a22
        add(a11, a22, s5, newSize);
        
        // s6 = b11 + b22
        add(b11, b22, s6, newSize);

        // s7 = a12 - a22
        sub(a12, a22, s7, newSize);

        // s8 = b21 + b22
        add(b21, b22, s8, newSize);

        // s9 = a11 - a21
        sub(a11, a21, s9, newSize);

        // s10 = b11 + b12
        add(b11, b12, s10, newSize);

        mat_multiply(s7, s8, p1, newSize, newSize, newSize);

        mat_multiply(s5, s6, p2, newSize, newSize, newSize);

        mat_multiply(s9, s10, p3, newSize, newSize, newSize);

        mat_multiply(s2, b22, p4, newSize, newSize, newSize);

        mat_multiply(a11, s1, p5, newSize, newSize, newSize);

        mat_multiply(a22, s4, p6, newSize, newSize, newSize);

        mat_multiply(s3, b11, p7, newSize, newSize, newSize);

        // c11 = p1 + p2 - p4 + p6
        add(p1, p2, tempA, newSize); // p1 + p2
        add(tempA, p6, tempB, newSize); // (p1 + p2) + p6
        sub(tempB, p4, c11, newSize); // (p5 + p4 + p6) - p2

        // c12 = p4 - p5
        sub(p1, p5, c12, newSize);

        // c21 = p6 + p7
        add(p6, p7, c21, newSize);

        // c22 = p2 - p3 + p5 - p7
        sub(p2, p3, tempA, newSize); // p2 - p3
        sub(tempA, p7, tempB, newSize); // (p2 - p3) - p7
        add(tempB, p5, c22, newSize); // (p2 - p3 - p7) + p5

        for (int i = 0; i < newSize ; i++) {
            for (int j = 0 ; j < newSize ; j++) {
                mult[idx(i,j,newSize)] = c11[idx(i,j,newSize)];
                mult[idx(i,(j + newSize),newSize)] = c12[idx(i,j,newSize)];
                mult[idx((i + newSize),j,newSize)] = c21[idx(i,j,newSize)];
                mult[idx((i + newSize),(j + newSize),d11)] = c22[idx(i,j,newSize)];
            }
        }
    }

    return;
}
