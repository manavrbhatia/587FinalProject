#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

int main()
{
    int a[10][10], b[10][10], mult[10][10], r1, c1, r2, c2, i, j, k;

    r1 = 4;
    c1 = 4;
    r2 = 4;
    c2 = 4;

    srand (time(NULL));

    for(i = 0; i < r1; ++i)
        for(j = 0; j < c1; ++j)
        {
            a[i][j] = rand() % 100;
        }

    for(i = 0; i < r2; ++i)
        for(j = 0; j < c2; ++j)
        {
            b[i][j] = rand() % 100;
        }

    // Initializing elements of matrix mult to 0.
    for(i = 0; i < r1; ++i)
        for(j = 0; j < c2; ++j)
        {
            mult[i][j]=0;
        }

    // Multiplying matrix a and b and storing in array mult.
    for(i = 0; i < r1; ++i)
        for(j = 0; j < c2; ++j)
            for(k = 0; k < c1; ++k)
            {
                mult[i][j] += a[i][k] * b[k][j];
            }

    // Displaying the multiplication of two matrix.
    cout << endl << "Output Matrix: " << endl;
    for(i = 0; i < r1; ++i)
    for(j = 0; j < c2; ++j)
    {
        cout << " " << mult[i][j];
        if(j == c2-1)
            cout << endl;
    }

    return 0;
}