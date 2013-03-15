#include "utils.h"

__global__
void d_gaussian_blur(const uchar4 *const inputImageRGBA,
                           uchar4 *const outputImageRGBA,
                     const int           imageHeight,
                     const int           imageWidth,
                     const float  *const filter,
                     const int           filterWidth)
{
    const int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int offset = pos.y * imageWidth + pos.x;

    if (pos.x >= imageWidth || pos.y >= imageHeight)
        return;


    // copy filter to shared memory
    extern __shared__ float shared_filter[];
    if (threadIdx.x < filterWidth && threadIdx.y < filterWidth)
    {
        const int ofs = threadIdx.x * filterWidth + threadIdx.y;
        shared_filter[ofs]   = filter[ofs];
    }
    __syncthreads();


    // TODO: prefetch image data for block + padding


    float3 result = make_float3(0,0,0);
    const int half_width = filterWidth / 2;

    for (int filter_y = -half_width; filter_y <= half_width; ++filter_y)
    for (int filter_x = -half_width; filter_x <= half_width; ++filter_x)
    {
        // clamping
        int image_y = min(max(pos.y + filter_y, 0), imageHeight-1);
        int image_x = min(max(pos.x + filter_x, 0), imageWidth-1);

        // calc. offsets
        int image_offset  = image_y * imageWidth + image_x;
        int filter_offset = (filter_y + half_width) * filterWidth + (filter_x + half_width);

        // fetch data
        uchar4 image_val = inputImageRGBA[image_offset];
        float filter_val = shared_filter[filter_offset];

        result.x += image_val.x * filter_val;
        result.y += image_val.y * filter_val;
        result.z += image_val.z * filter_val;
    }

    outputImageRGBA[offset] = make_uchar4(result.x, result.y, result.z, 255);
}


void gaussian_blur(const uchar4 *const d_inputImageRGBA,
                         uchar4 *const d_outputImageRGBA,
                   const int           imageHeight,
                   const int           imageWidth,
                   const float  *const d_filter,
                   const int           filterWidth)
{
    const int numThreads = 16;

    const dim3 numBlocks(imageWidth/numThreads+1, imageHeight/numThreads+1);
    const dim3 threadsPerBlock(numThreads, numThreads);

    const size_t filterSize = sizeof(float) * filterWidth * filterWidth;

    d_gaussian_blur<<<numBlocks, threadsPerBlock, filterSize>>>
                   (d_inputImageRGBA, d_outputImageRGBA,
                    imageHeight, imageWidth,
                    d_filter, filterWidth);

    // cudaDeviceSynchronize();
    // checkCudaErrors(cudaGetLastError());
}