#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cassert>
#include <omp.h>
//#include "naive_mm.cpp"
#include "omp_naive.cpp"

using namespace std;

double f_a(int i,int j) {
    return 1;
    return rand() % 100;
}

double f_b(int i,int j) {
    return 1;
    return rand() % 100;
}

int main()
{
    srand (time(NULL));

    int i,j;
    int r1 = 4;
    int c1 = 4;
    double a[r1*c1];

    int r2 = 4;
    int c2 = 4;
    double b[r2*c2];
    
    double* mult = (double *) malloc(r1*c2*sizeof(double)); 

    assert(c1==r2);

    // Initialize a and b using functionis of your choice
    for(i = 0; i < r1; ++i)
        for(j = 0; j < c1; ++j)
            a[idx(i,j,r1)] = f_a(i,j);

    for(i = 0; i < r2; ++i)
        for(j = 0; j < c2; ++j)
            b[idx(i,j,r2)] = f_b(i,j);

    // Multiplying matrix a and b and storing in array mult.
    naive_mm(a,b,mult,r1,c1,c2);

    // Displaying the multiplication of two matrix.
    cout << endl << "Output Matrix: " << endl;
    for(i = 0; i < r1; ++i)
        for(j = 0; j < c2; ++j)
        {
            cout << " " << mult[idx(i,j,r1)];
            if(j == c2-1)
                cout << endl;
        }

    delete mult;

    return 0;
}
