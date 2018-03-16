// Blocks and Threads
// blockIdx.x->        0     |      1    |     2
// threadIdx.x-> [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5]
//
// blockDim -> Block dimension = number of threads per block
//
// index = threadIdx + (blockIdx.x * M)

#include <stdio.h>
#include <time.h>
#define N 2048*2048
#define M 512 // THREADS_PER_BLOCK

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n) // avoid accessing beyond end of array, when not perfect multiples.
        c[index] = a[index] + b[index];
}

void random_ints(int *a, int n){
   int i;
   for (i = 0; i < n; ++i)
        a[i] = rand()%100;
}

int main(void) {
    int *a, *b, *c; // host copy
    int *d_a, *d_b, *d_c; // device copy
    int size = N * sizeof(int);

    // allocate mem for device copies
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // copy inputs to device
    // cudaMemcpy(destination, source, size, direction);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on GPU with N threads
    add<<<(N + M-1)/M, M>>>(d_a, d_b, d_c, N);

    // copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("exec time: %f seconds\n", cpu_time_used);

    // cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
