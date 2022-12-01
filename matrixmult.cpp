#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cassert>
#include <math.h>
#include "f.h"
#include <cstdlib>
#include <cstring>


//#include "naive_mm.cpp"
//#include "omp_naive.cpp"
//#include "CUDA_mm.cu"
//#include "mpi_strassens.cpp"
#include "strassens_cpu.cpp"
#define idx(x,y,M) (M*(x)+(y))

using namespace std;

double f_a1(int i, int j, int size) {
    return i*sin(i)+j*cos(j)+sqrt(i+j);
}

double f_a2(int i, int j, int size) {
    return idx(i,j,size) % (size/2) == 1 ? sin(idx(i,j,size)):0;
}

double f_a(int i, int j, int size) {
    return i*sin(i)+j*cos(j)+sqrt(i+j);
    return idx(i,j,size) % (size/2) == 1 ? sin(idx(i,j,size)):0;
}

double f_b1(int i, int j, int size) {
    return j*sin(j)+i*cos(i)+(i+j)*(i+j);
}

double f_b2(int i, int j, int size) {
    return idx(i,j,size) % (size/2) == 1 ? sin(idx(i,j,size)):0;
}

double f_b(int i, int j, int size) {
    return j*sin(j)+i*cos(i)+(i+j)*(i+j);
    return idx(i,j,size) % (size/2) == 1 ? sin(idx(i,j,size)):0;
}

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 3)
    {
        cout << "Please Give a matrix size and function to use";
        return -1;
    }
    cout << "Size is " << atoi(argv[1]) << " Function is " << argv[2] << endl;

    srand (time(NULL));

    int i,j;
    if (atoi(argv[1]) != 256 && atoi(argv[1]) != 2048)
    {
        cout << "Please Give a matrix size that is valid (256 or 2000)";
        return -1;
    }
    int r1 = atoi(argv[1]);
    int c1 = r1;
    double a[r1*c1];

    int r2 = c1;
    int c2 = r2;
    double b[r2*c2];
    
    double* mult = (double *) malloc(r1*c2*sizeof(double)); 

    assert(c1==r2);

    // Initialize a and b using functionis of your choice
    if (strcmp(argv[2], "f1")==0)
        cout << "Using function 1" << endl;
    else if(strcmp(argv[2], "f2")==0)
        cout << "Using function 2" << endl;
    else
    {
        cout << "Using custom function" << endl;
    }
    for(i = 0; i < r1; ++i)
        for(j = 0; j < c1; ++j)
        {
            if (strcmp(argv[2], "f1")==0)
                a[idx(i,j,r1)] = f_a1(i,j,r1);
            else if(strcmp(argv[2], "f2")==0)
                a[idx(i,j,r1)] = f_a2(i,j,r1);
            else
            {
                a[idx(i,j,r1)] = f_a(i,j,r1);
            }
        }

    for(i = 0; i < r2; ++i)
        for(j = 0; j < c2; ++j)
        {
            if (strcmp(argv[2], "f1")==0)
                b[idx(i,j,r2)] = f_b2(i,j,r2);
            else if(strcmp(argv[2], "f2")==0)
                b[idx(i,j,r2)] = f_b2(i,j,r2);
            else
            {
                b[idx(i,j,r2)] = f_b(i,j,r2);
            }
        }

    // Multiplying matrix a and b and storing in array mult.
    mat_multiply(a,b,mult,r1,c1,c2);

    double sum = 0;
    double sq_sum = 0;
    // Displaying the multiplication of two matrix.
    cout << endl << "Output Matrix: " << endl;
    for(i = 0; i < r1; ++i)
        for(j = 0; j < c2; ++j)
        {
            sum += mult[idx(i,j,r1)];
            sq_sum += mult[idx(i,j,r1)]*mult[idx(i,j,r1)];
        }

    printf("\nThe sum retrieved was %f and the square sum was %f",sum,sq_sum); 
    delete mult;

    return 0;
}
