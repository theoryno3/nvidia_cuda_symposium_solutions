#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <stdio.h>

/* macro for index calculations */

#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )

/* matrix size and thread dimensions */

#define SIZE 1024

/* setup various hard-coded parameters for this kernel */

#define TBX 64 // Size of C this CTA is responsible for, x dimension
#define TBY 64 // Size of C this CTA is responsible for, y dimension
#define THREADS_PER_BLOCK_X 16 // Thread block size, x dimension
#define THREADS_PER_BLOCK_Y 16 // Thread block size, y dimension
#define BK 16 // square block of K size
#define NX 4  // = TBX/THREADS_PER_BLOCK_X == number of iterations to do TBX work with THREADS_PER_BLOCK_X blocks
#define NY 4  // = TBY/THREADS_PER_BLOCK_Y == number of iterations to do TBY work with THREADS_PER_BLOCK_Y blocks

__global__ void GPU_shmem1(const int m, double const * const a, double const * const b, double *c )
{

/* setup some constants for later use */

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int iby = blockIdx.y * TBY;
	const int ibx = blockIdx.x * TBX;

/* shared memory arrays for A and B */

	__shared__ double as[ TBX ][ BK+1 ];
	__shared__ double bs[ BK ][ TBY+1 ];
	
/* space for C to be held in registers */

	double c_tmp[ NX ][ NY ] ;

	/* zero the temp C array */

#pragma unroll
	for ( int i = 0 ; i < NX ; i++) { 
		for ( int j = 0 ; j < NY ; j++) {
			c_tmp[i][j] = 0.0;
		}
	}

	/* calculate my initial offset into A and B */

	int aoff = INDX( ibx + tx, ty, m );
	int boff = INDX( tx, iby + ty, m );

	/* main loop over blocks of K */

	for( int Kblock = 0; Kblock < m; Kblock+=BK )
	{

		/* read block of A into shared memory */

#pragma unroll
		for ( int i = 0; i < NX ; i ++ ) 
		{
			as[ tx + i * THREADS_PER_BLOCK_X ][ ty ] = a[ (aoff + i*THREADS_PER_BLOCK_X) ];
		}

		/* read block of B into shared memory */

#pragma unroll
		for ( int i = 0; i < NY ; i ++ ) 
		{
			bs[ tx ][ ty + THREADS_PER_BLOCK_Y * i ] = b[ (boff + m*i*THREADS_PER_BLOCK_Y) ];
		}

		__syncthreads();

		/* increment A and B offsets  for next round of data reads */

		boff += BK;
		aoff += m * BK;

		/* triply nested loop to perform the matmult on the blocks */

#pragma unroll
		for( int k = 0 ; k < BK ; k++ )
		{
#pragma unroll
			for (int j = 0 ; j < NY ; j++ )
			{
#pragma unroll
				for (int i = 0 ; i < NX ; i++ )
				{
					c_tmp[ i ][ j ] += as[ tx + THREADS_PER_BLOCK_X*i ][ k ] * bs[ k ][ ty + j*THREADS_PER_BLOCK_Y ];
				}
			}
		}
		__syncthreads();

	} /* end for Kblock */

	/* set coff to its proper index int the C matrix */

	int coff = INDX( ibx + tx, iby + ty, m );
  
	/* write results to the C matrix */

#pragma unroll
	for ( int j = 0 ; j < NY ; j++ ) 
	{
#pragma unroll
		for ( int i = 0 ; i < NX ; i++ )
		{      
			c[ coff + INDX( THREADS_PER_BLOCK_X * i, THREADS_PER_BLOCK_Y * j, m )] = c_tmp[i][j];
		}
	}
 
} /* end GPU_shmem1 */

int main( int argc, char *argv[] )
{

    const int size = SIZE;

    fprintf(stdout, "Matrix size is %d\n",size);

    double *h_a, *h_b, *h_c, *h_c1;
    double *d_a, *d_b, *d_c;
 
    size_t numbytes = (size_t ) size * (size_t ) size * sizeof( double );

    h_a = (double *) malloc( numbytes );
    if( h_a == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_b = (double *) malloc( numbytes );
    if( h_b == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

    h_c = (double *) malloc( numbytes );
    if( h_c == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

	h_c1 = (double *) malloc( numbytes );
    if( h_c1 == NULL )
    {
      fprintf(stderr,"Error in host malloc\n");
      return 911;
    }

	/* zero out the host memory for C matrices */

    memset( h_c, 0, numbytes );
    memset( h_c1, 0, numbytes );

    fprintf( stdout, "Total memory required is %lf MB\n", 
       3.0 * (double) numbytes / 1000000.0 );

	/* initialize the A and B matrices */

    for( int i = 0; i < size * size; i++ )
    {
      h_a[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
      h_b[i] = double( rand() ) / ( double(RAND_MAX) + 1.0 );
    }

	/* allocate a, b, c in gpu memory */

    cudaMalloc( (void **)&d_a, numbytes );
    cudaMalloc( (void **)&d_b, numbytes );
    cudaMalloc( (void **)&d_c, numbytes );
	
	/* copy a and b to device */

	cudaMemcpy( d_a, h_a, numbytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b, numbytes, cudaMemcpyHostToDevice );

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate( &handle );

    double alpha = 1.0;
    double beta  = 0.0;

	/* start timers */

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

	/* call CUBLAS dgemm */

cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 size, size, size,
                 &alpha, 
                 d_a, size,
                 d_b, size,
                 &beta,
                 d_c, size );

	/* stop timers */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

	/* print GPU CUBLAS timing information */

    fprintf(stdout, "Total time GPU CUBLAS is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
    
	/* copy C from device to host for error checking */

    cudaMemcpy( h_c, d_c, numbytes, cudaMemcpyDeviceToHost );

	/* reset C on device to zero */

	cudaMemset( d_c, 0, numbytes );

	/* setup grid and block sizes */

	dim3 blocksize( THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1 );
	dim3 gridsize( size / ( TBX ), size / ( TBY ), 1 );	

	/* call GPU_naive */

	printf("block.X %d block.Y %d\n",gridsize.x, gridsize.y );
	printf("threads.x %d threads.y %d\n",blocksize.x, blocksize.y );
    
/* start timers */

	cudaEventRecord( start, 0 );

/* call the kernel */

	GPU_shmem1<<< gridsize, blocksize >>> ( size, d_a, d_b, d_c );

	/* stop timers */

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
	elapsedTime = 0.0f;
    cudaEventElapsedTime( &elapsedTime, start, stop );

	/* print data for GPU naive */

    fprintf(stdout, "Total time GPU SHMEM is %f sec\n", elapsedTime / 1000.0f );
    fprintf(stdout, "Performance is %f GFlop/s\n", 
      2.0 * (double) size * (double) size * (double) size / 
      ( (double) elapsedTime / 1000.0 ) * 1.e-9 );
                  
	/* copy C back to host */
	
	cudaMemcpy( h_c1, d_c, numbytes, cudaMemcpyDeviceToHost );

    cublasDestroy( handle );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

	/* check CUBLAS versus GPU NAIVE numerical results */

	double temp = 0.0;

	for( int i = 0; i < size * size; i++ )
	{
		temp += ( h_c[i] - h_c1[i] ) * ( h_c[i] - h_c1[i] );
      //  printf("i %d h_c %f h_c1 %f\n",i,h_c[i],h_c1[i] );
	} /* end for */

	printf("error is %f\n",temp);
	if( temp > 10 ) printf("Error value is suspiciously high!\n");

	/* cleanup */

    cudaFree( d_a );
    cudaFree( d_b );
	cudaFree( d_c );

    free( h_a );
    free( h_b );
    free( h_c );
    free( h_c1 );

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
