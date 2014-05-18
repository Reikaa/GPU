GPU
===

This project tackled a simulation of image processing using the GPU. The problem was how to handle matrices up to the size 4096 * 4096 within a reasonable amount of time and the solution was to use the GPU. What the code does is handle .bmp images and compares each bit to the same corresponding bit in a template to determine if the image is what the template matches. It can also determine if two matrices are similar or if a matrix contains a subset of a smaller matrix. Below is a sample of the reduction kernal I chose to use to create a faster reduction of euclidean distances calculated. It is combined with loop unroll to increase GFlop performance. 

#Reduction Kernel & Loop Unroll

```cpp

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

```

#Minimum Distance Calculator

```cpp

float calc_min_dist(float *image, int i_width, int i_height, float *temp, int t_width) {
    // float* image and float* temp are pointers to GPU addressible memory
    // You MAY NOT copy this data back to CPU addressible memory and you MAY 
    // NOT perform any computation using values from image or temp on the CPU.
    // The only computation you may perform on the CPU directly derived from distance
    // values is selecting the minimum distance value given a calculated distance and a 
    // "min so far"

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
```
