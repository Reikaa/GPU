/*
 * Proj 3-2 SKELETON
 */

/* Felix Liu and Brian Truong */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>
// #include <emmintrin.h> /* This allows SSE Instrinsics to work. */
#include <cutil.h>
#include "utils.h"

/* Does a horizontal flip for an array using the GPU. */
__global__ void flipKernal(float *arr, int width) {
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (x < width/2) {
        float tmp = arr[(y * width) + (width - x - 1)];
        arr[(y * width) + (width - x - 1)] = arr[(width * y + x)];
        arr[(width * y + x)] = tmp;
	}
}

/* Transpose Kernal for GPU use. */
__global__ void transposeKernal(float *arr, int width) {
	int y = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (y < width && x < width) {
        if (x != y && (y * width + x) < (x * width + y)) {
            float tmp = arr[y * width + x];
            arr[y * width + x] = arr[x * width + y];
            arr[x * width + y] = tmp;
        }
	}
}

/* Euclid Kernal for GPU to find calculations of the euclidian distance. */
__global__ void euclidKernal(float* image, float* temp, int row, int column, int i_width, int t_width, float* hold) {
    int row_in = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_in < t_width && col < t_width) {
		uint temp_calc = row_in + col * t_width;
        uint image_calc = row_in + row + (col * i_width) + (column * i_width);
        float unsquared = image[image_calc] - temp[temp_calc];
        float actual = unsquared * unsquared;
        hold[temp_calc] = actual;
    }
}

/* Unrolling for the final piece of data. */
__device__ void unrollReduce(volatile float *hold, long thisThreadIndex) {
    hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + (long)32];
    hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + (long)16];
    hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + (long)8];
    hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + (long)4];
    hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + (long)2];
    hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + (long)1];
}

/* Reduction Kernal for GPU for maximum reduction speed. */
__global__ void reductionKernal(float *hold, long len, long level) {
    long thisThreadIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thisThreadIndex + level < len && level > 32) {
        hold[thisThreadIndex] = hold[thisThreadIndex] + hold[thisThreadIndex + level];
	} else if (level <= 32) {
		unrollReduce(hold, thisThreadIndex);
	}
}

/* Returns the squared holdidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */
float calc_min_dist(float *image, int i_width, int i_height, float *temp, int t_width) {
    // float* image and float* temp are pointers to GPU addressible memory
    // You MAY NOT copy this data back to CPU addressible memory and you MAY 
    // NOT perform any computation using values from image or temp on the CPU.
    // The only computation you may perform on the CPU directly derived from distance
    // values is selecting the minimum distance value given a calculated distance and a 
    // "min so far"
    /* YOUR CODE HERE */


    float min_dist = FLT_MAX;
	int dim_of_template = t_width * t_width;
    size_t arraySize = dim_of_template*sizeof(float); 
	float *hold;
    CUDA_SAFE_CALL(cudaMalloc(&hold, arraySize));

    dim3 dim_blocks_per_grid(t_width/4, t_width/4);
    dim3 dim_threads_per_block(4, 4, 1);

    for (int i = 0; i < 8; i++) {
        if (i % 2 == 0 && i != 0) {
            flipKernal<<<dim_blocks_per_grid, dim_threads_per_block>>>(temp, t_width);
            cudaThreadSynchronize();
            CUT_CHECK_ERROR("");
        } else if (i != 0) {
            transposeKernal<<<dim_blocks_per_grid, dim_threads_per_block>>>(temp, t_width);
            cudaThreadSynchronize();
            CUT_CHECK_ERROR("");
        }
        for (int row = 0; row <= (i_height - t_width); row++) {
            for (int column = 0; column <= (i_width - t_width); column++) {
                euclidKernal<<<dim_blocks_per_grid, dim_threads_per_block>>>(image, temp, column, row, i_width, t_width, hold);
                cudaThreadSynchronize();
                CUT_CHECK_ERROR("");

                long threads_per_block = 512;
                long blocks_per_grid = ((dim_of_template) / threads_per_block) + 1;
                long len = dim_of_template;
                long level = len/2;
                while (level > 0) {
                    dim3 dim_blocks_per_grid(blocks_per_grid, 1);
                    dim3 dim_threads_per_block(threads_per_block, 1, 1);
                    reductionKernal<<<dim_blocks_per_grid, dim_threads_per_block>>>(hold, len, level);
                    cudaThreadSynchronize();
                    CUT_CHECK_ERROR("");
                    if (level > 32) {
	                    level >>= 1;
                    } else {
                    	level = 0;
                    }
                    if (blocks_per_grid != 1) {
                        blocks_per_grid = (blocks_per_grid / 2);
                    }
                }
                float check;
                CUDA_SAFE_CALL(cudaMemcpy(&check, hold, sizeof(float), cudaMemcpyDeviceToHost));
                if (check < min_dist) {
                    min_dist = check;
                }
            }
        }
    }
    CUDA_SAFE_CALL(cudaFree(hold));
    return min_dist;
}
