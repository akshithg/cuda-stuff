// Threads
// threadIdx -> thread index

#include <stdio.h>
#define N 512

__global__ void add(int *a, int *b, int *c) {
    // use threadIdx.x to access thread index
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
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
    int i;

    // allocate mem for device copies
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    // copy inputs to device
    // cudaMemcpy(destination, source, size, direction);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on GPU with N threads
    add<<<1, N>>>(d_a, d_b, d_c);

    // copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for(i=0; i<N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
