// Blocks
// blockIdx -> block index

#include <stdio.h>
#include <time.h>
#define N 2048*2048

__global__ void add(int *a, int *b, int *c) {
    // use blockIdx.x to access block index
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
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

    // launch add() on GPU
    // instead of executing add once, execute N time in parallel
    add<<<N, 1>>>(d_a, d_b, d_c);

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
