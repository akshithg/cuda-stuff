// Memory Allocation
// Simple op to demonstrate memory allocation
// malloc, free, memcpy = cudaMalloc, cudaFree, cudaMemcpy

#include<stdio.h>

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main(void) {
    int a, b, c; // host copy
    int *d_a, *d_b, *d_c; // device copy
    int size = sizeof(int);

    // allocate mem for device copies
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    a = 1;
    b = 5;

    // copy inputs to device
    // cudaMemcpy(destination, source, size, direction);
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // launch add() on GPU
    add<<<1, 1>>>(d_a, d_b, d_c);

    // copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("%d + %d = %d\n", a, b, c);

    return 0;
}
