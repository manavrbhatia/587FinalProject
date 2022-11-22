#define TILE_WIDTH 16; 

__global__ void naive_mult(double *A, double *B, double *C, int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < size) && (col  < size)){
        double value = 0;

        for(int k = 0; k < size; k++){
            value += A[row*size + k] * B[k*size + col]; 
        }
        C[row*size + col] = value; 
    }
}

__global__ void naive_mult_tile(double *A, double *B, double *C, int size){
    __shared__ subA[TILE_WIDTH][TILE_WIDTH];
    __shared__ subB[TILE_WIDTH][TILE_WIDTH];
    
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow*TILE_WIDTH + blockCol; 
    int col = threadRow*TILE_WIDTH + thead_col; 

    double value = 0;
    for(int sub_i = 0; sub_i < width/TILE_WIDTH; sub_i++) {
        subA[threadRow][threadCol] = A[idx(row, sub_i*TILE_WIDTH+threadCol, width)];
        subB[threadRow][threadCol] = B[idx(i*TILE_WIDTH+threadRow, col, width)];
        __syncthreads();
   
        for(int k = 0; k < TILE_WIDTH; k++){
            value += subA[threadRow][k] * subB[k][threadCol]; 
        }
        _synchthreads();
    }
    C[row*size + col] = value; 
}

double f_a(int i,int j) {
    return 1;
    return rand() % 100;
}

double f_b(int i,int j) {
    return 1;
    return rand() % 100;
}

void mat_multiply(double* A, double* B, double* C, int d11, int d12, int d22) {
    // Initializing elements of matrix mult to 0.
    for(int i = 0; i < d11; ++i)
        for(int j = 0; j < d22; ++j)
            mult[idx(i,j,d11)]=0;
    
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

    dim3 tbp(20,20);
    dim3 numBlocks(N/tbp.x,N/tbp.y);

    cudaEventRecord(time_s);
    for (int step = 0; step < t; step++){
        naive_mult <<< numBlocks, tbp >>> (dA, dB, dC, d12);
    }
    cudaEventRecord(time_e);
    cudaEventSynchronize(time_e);

    printf("Time the function took is %.4fs", cudaEventElapsedTime(&time, &time_s, &time_e))
    cudaMemcpy(C,dC,(d11*d22)*(sizeof(double)),cudaMemcpyDeviceToHost);
    // Multiplying matrix a and b and storing in array mult.

    return;
}

int main()
{
    srand (time(NULL));
    cudaEvent_t time_s, time_e;
    cudaEventCreate(&time_s);
    cudaEventCreate(&time_e);


    int i,j;
    int r1 = 4;
    int c1 = 4;

    int r2 = 4;
    int c2 = 4;
    
    double *dA, *dA_0;
    double *A = (double*) malloc((r1*c1)*sizeof(double));
    double *B = (double*) malloc((r2*c2)*sizeof(double));
    double *c = (double*) malloc(r1*c2*sizeof(double)); 

    cudaMalloc((void**)&dA,(r1*c1)*sizeof(double));
    cudaMalloc((void**)&dB,(r2*c2)*sizeof(double));
    cudaMalloc((void**)&C,(r2*c2)*sizeof(double));

    assert(c1==r2);

    // Initialize a and b using functionis of your choice
    for(int i = 0; i < r1; ++i)
        for(int j = 0; j < c1; ++j)
            a[idx(i,j,r1)] = f_a(i,j);

   for(int i = 0; i < r2; ++i)
        for(int j = 0; j < c2; ++j)
            b[idx(i,j,r2)] = f_b(i,j);

    // Multiplying matrix a and b and storing in array mult.
    mat_multiply(a,b,mult,r1,c1,c2);

    // Displaying the multiplication of two matrix.
    cout << endl << "Output Matrix: " << endl;
    for(int i = 0; i < r1; ++i)
        for(int j = 0; j < c2; ++j)
        {
            cout << " " << mult[idx(i,j,r1)];
            if(j == c2-1)
                cout << endl;
        }

    delete mult;

    return 0;
}