
#include "cuda_runtime.h"
#include <stdio.h>
#include <math.h>

/* definitions of threadblock size in X and Y directions */

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

/* definition of matrix linear dimension */

#define SIZE 1024

/* macro to index a 1D memory array with 2D indices in column-major order */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* CUDA kernel for shared memory matrix transpose */

__global__ void smem_cuda_transpose( const int m, double const * const a, double *c )
{
	
/* declare a shared array */

	__shared__ double smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];
	
/* determine my row and column indices */

	const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
	const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

/* determine my row block and column block indices */

	const int sourceBlockX = blockDim.x * blockIdx.x;
	const int sourceBlockY = blockDim.y * blockIdx.y;

	if( myRow < m && myCol < m )
	{
		
		/* read to the shared mem array */
        smemArray[threadIdx.x][threadIdx.y] = a[INDX( sourceBlockX + threadIdx.x, sourceBlockY + threadIdx.y, m )];
		
		/* synchronize */
		__syncthreads();
		
		/* write the result */
	    c[INDX( sourceBlockY + threadIdx.x, sourceBlockX + threadIdx.y, m )] = smemArray[threadIdx.y][threadIdx.x];
	
	} /* end if */
	return;
} /* end naive_cuda_transpose */

void host_transpose( const int m, double const * const a, double *c )
{
	
/* 
 *  naive matrix transpose goes here.
 */
 
 for( int j = 0; j < m; j++ )
	{
		for( int i = 0; i < m; i++ )
		{
		    c[INDX(i,j,m)] = a[INDX(j,i,m)];
		} /* end for i */
	} /* end for j */

} /* end host_dgemm */

int main( int argc, char *argv[] )
{

    int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

/* declaring pointers for array */

    double *h_a, *h_c;
    double *d_a, *d_c;
 
    size_t numbytes = (size_t) size * (size_t) size * sizeof( double );

/* allocating host memory */

    h_a = (double *) malloc( numbytes );
    if( h_a == NULL )
    {
      fprintf(stderr,"Error in host malloc h_a\n");
      return 911;
    }

    h_c = (double *) malloc( numbytes );
    if( h_c == NULL )
    {
      fprintf(stderr,"Error in host malloc h_c\n");
      return 911;
    }

/* allocating device memory */

    cudaMalloc( (void**) &d_a, numbytes );
    cudaMalloc( (void**) &d_c, numbytes );

/* set result matrices to zero */

    memset( h_c, 0, numbytes );
    cudaMemset( d_c, 0, numbytes );

    fprintf( stdout, "Total memory required per matrix is %lf MB\n", 
       (double) numbytes / 1000000.0 );

/* initialize input matrix with random value */

    for( int i = 0; i < size * size; i++ )
    {
      //h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
       h_a[i] = (double)i;
	}

/* copy input matrix from host to device */

    cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice );

/* create and start timer */

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

/* call naive cpu transpose function */

    host_transpose( size, h_a, h_c );

/* stop CPU timer */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

/* print CPU timing information */

    fprintf(stdout, "Total time CPU is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GB/s\n", 
      8.0 * 2.0 * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* setup threadblock size and grid sizes */

    dim3 blocksize( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
    dim3 gridsize( ( size / blocksize.x ) + 1, ( size / blocksize.y ) + 1, 1 );

/* start timers */
    cudaEventRecord( start, 0 );

/* call naive GPU transpose kernel */

    smem_cuda_transpose<<< gridsize, blocksize >>>( size, d_a, d_c );

/* stop the timers */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );

/* print GPU timing information */

    fprintf(stdout, "Total time GPU is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GB/s\n", 
      8.0 * 2.0 * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );

/* copy data from device to host */

    memset( h_a, 0, numbytes );
    cudaMemcpy( h_a, d_c, numbytes, cudaMemcpyDeviceToHost );

/* compare GPU to CPU for correctness */

	for( int j = 0; j < size; j++ )
	{
		for( int i = 0; i < size; i++ )
		{
		    if( h_c[INDX(i,j,size)] != h_a[INDX(i,j,size)] ) 
                    {
                      printf("Error in element %d,%d\n", i,j );
                      printf("Host %f, device %d\n",h_c[INDX(i,j,size)],
                                                    h_a[INDX(i,j,size)]);
                    }
		} /* end for i */
	} /* end for j */

/* free the memory */

    free( h_a );
    free( h_c );
    cudaFree( d_a );
    cudaFree( d_c );

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
