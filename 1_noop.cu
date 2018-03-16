// Noop
// Device code that does nothing

#include<stdio.h>

__global__ void mykernel(void) { // this runs on device
}

int main(void) {
    mykernel<<<1, 1>>>();
    printf("Hello! \n");
    return 0;
}
