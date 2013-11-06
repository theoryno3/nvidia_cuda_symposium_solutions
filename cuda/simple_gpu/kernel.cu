
#include "cuda_runtime.h"
#include <stdio.h>

int main()
{
    
	int dimx = 16;
	int num_bytes = dimx * sizeof( int ); 

	int *d_a = 0, *h_a = 0; //device and host pointers

	h_a = (int *) malloc( num_bytes );
	// allocate memory on the GPU
	cudaMalloc( (void **) &d_a, num_bytes );

	if( 0 == h_a || 0 == d_a )
	{  
		printf("couldn't allocate memory\n");
		return 911;
	} /* end if */

	// memset on the gpu
	cudaMemset( d_a, 0, num_bytes );
	//
	cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );

	for( int i = 0; i < dimx; i++ )
	{
		printf("%d ", h_a[i] );
	}
	printf("\n");

	free( h_a );
	cudaFree( d_a );

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    if ( cudaDeviceReset() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}