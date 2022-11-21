#include <time.h>
#include <omp.h>
#include <bits/stdc++.h>

#define idx(x,y,M) (M*(x)+y)
using namespace std;

/* Performs naive matrix multiplication using given matrices and their dimension.
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

void mat_multiply(double* a, double* b, double* mult, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.
    printf("Number of threads: %d\n", omp_get_max_threads());

    for(int i = 0; i < d11; ++i)
        for(int j = 0; j < d22; ++j)
            mult[idx(i,j,d11)]=0;

    double start = omp_get_wtime(); 
    // Multiplying matrix a and b and storing in array mult.
    #pragma omp parallel for 
    for(int i = 0; i < d11; ++i)
        for(int j = 0; j < d22; ++j)
            for(int k = 0; k < d12; ++k)
                mult[idx(i,j,d11)] += a[idx(i,k,d11)] * b[idx(k,j,d12)];
    double end = omp_get_wtime(); 

    printf("Took %f seconds to perform matrix multiplication.\n", end-start);
    return;
}
