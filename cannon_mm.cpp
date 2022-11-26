#include <mpi.h>
#include <math.h> 
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;

#define n 400
#define idx(x,y,M) (M*(x)+(y))

int main(int argc, char**argv) { 
	MPI_Init(&argc, &argv); 
	int p;
	MPI_Comm_size(MPI_COMM_WORLD, &p);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // Dimensions of the square grid of procs
    int p_rt = sqrt(p);

    // Get row and column of processor in virtual grid
    int p_i = id/p_rt;
    int p_j = id%p_rt;

    // Submatrix size
    int N = ceil(n/p_rt);

    // Initialize the root A and B (and C), then distribute it;
    // TODO: Change this to read from a file or use a function
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    if (id == 0) {
        A = (double*) malloc(n*n*sizeof(double));
        B = (double*) malloc(n*n*sizeof(double));
        C = (double*) malloc(n*n*sizeof(double));

        for(int i = 0; i<n*n;i++) {
            A[i] = 1;
            B[i] = 1;
        }
    }

	// Start time on root processor
	double startt;
    if (id == 0) startt = MPI_Wtime();

    // Make the shared datatype for the chunk layour we want, it's magic (z-ordering i believe)
    MPI_Datatype og_size, resized;
    int shape[2] = {n,n};
    int subshape[2] = {N,N};
    int start[2] = {0,0};

    MPI_Type_create_subarray(2, shape, subshape, start, MPI_ORDER_C, MPI_DOUBLE, &og_size);
    MPI_Type_create_resized(og_size, 0, N*sizeof(double), &resized);
    MPI_Type_commit(&resized);

    int counts[p];
    int disps[p];
    for(int i = 0; i<p;i++) {
        counts[i] = 1;
        disps[i] = (i%p_rt)+(i/p_rt)*p_rt*N;
    }

	// Initialize A,B main pointers
    double* A_sub = (double*) malloc(N*N*sizeof(double));
    double* B_sub = (double*) malloc(N*N*sizeof(double));
    double* C_sub = (double*) malloc(N*N*sizeof(double));

    // Distribute the first chunk according to processor location in virtual grid
    // Initialize sub_C to hold result chunk of C
    MPI_Scatterv(A, counts, disps, resized, A_sub, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, counts, disps, resized, B_sub, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i<N*N; i++) C_sub[i]=0;

    for(int i = 0; i < p_rt; i++){
        // Serial submatrix * and submatrix + ops on processor
        for (int a=0;a<N;a++) {
            for (int b=0;b<N;b++) {
                for (int c=0;c<N;c++) {
                    C_sub[idx(a,b,N)] += A_sub[idx(a,c,N)]*B_sub[idx(c,b,N)];
                }
            }
        }

        // Send to your current chunk to previous processor, in ring format
        MPI_Send(A_sub, N*N, MPI_DOUBLE, idx(p_i,(p_j+p_rt-1)%p_rt,p_rt), 0, MPI_COMM_WORLD);
        MPI_Send(B_sub, N*N, MPI_DOUBLE, idx((p_i+p_rt-1)%p_rt,p_j,p_rt), 0, MPI_COMM_WORLD);

        // Receive from next processor, in ring format; Blocking?
        MPI_Recv(A_sub, N*N, MPI_DOUBLE, idx(p_i,(p_j+1)%p_rt,p_rt), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B_sub, N*N, MPI_DOUBLE, idx((p_i+1)%p_rt,p_j,p_rt), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

	// Variables to store sum and square sum
	double n_sum= 0.0, s_sum= 0.0;

    // TODO: Add a way to retrieve the complete C from this procedure
	for(int i = 0; i<N; i++){
		for(int j=0; j<N; j++){
			double val = C_sub[idx(i,j,N)];
            cout << val <<endl;
			n_sum += val;
			s_sum += val*val;
		}
	}

	// Sum everything
	double sum_collect = 0.0;
	double s_sum_collect = 0.0;

	MPI_Reduce(&n_sum, &sum_collect, 1, MPI_DOUBLE,  MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&s_sum, &s_sum_collect, 1, MPI_DOUBLE,  MPI_SUM, 0, MPI_COMM_WORLD);
	if (id == 0) {
		double endt = MPI_Wtime();
		cout << "Total time elapsed:" << endt-startt << ";  Sum=" << sum_collect << "; Square Sum="<< s_sum_collect << endl;
	}

	// Clean up A and B completely
    if (id==0) {
        free(A);
        free(B);
        free(C);
    }

    free(A_sub);
    free(B_sub);
    free(C_sub);

	MPI_Finalize();
	return 0;
}
