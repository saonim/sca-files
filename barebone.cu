#include <stdio.h>

const int N = 1; 
const int blocksize = 1; 

__global__ void kernelFunc() {

}

int main() {
  int b[N] = {4};
  int *bd;
  const int isize = N*sizeof(int);
  printf("%i", *b);
  cudaMalloc( (void**)&bd, isize ); 
  cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 

  // Allocate a big chunk of memory as a trigger
  const int cnst = 1000000000;
  int *d_ptr;
  cudaMalloc(&d_ptr, cnst * sizeof(int));

  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  kernelFunc<<<dimGrid, dimBlock>>>();
  cudaMemcpy( b, bd, isize, cudaMemcpyDeviceToHost ); 
  cudaFree( bd );
  cudaFree( d_ptr );
  printf(" %i\n", *b);
  return EXIT_SUCCESS;
}
