#include <opencv2/opencv.hpp>
#include <cuda.h>

#include <iostream>
#include <string>
#include <stdio.h>

#include "utils.h"
#include "gaussian_blur.h"


float* d_allocFilter(const float *const h_filter,
                     const int          filterWidth)
{
    float *d_filter;

    const size_t filterSize = sizeof(float) * filterWidth * filterWidth;

    checkCudaErrors( cudaMalloc(&d_filter, filterSize) );
    checkCudaErrors( cudaMemcpy(d_filter, h_filter, filterSize,
                                cudaMemcpyHostToDevice) );
    return d_filter;
}


float* generateBlurKernel(const int   width,
                          const float sigma)
{
    float *h_filter  = new float[width * width];
    float  filterSum = 0.f; 
    int    hWidth    = width / 2;

    for (int y = -hWidth; y <= hWidth; ++y)
    for (int x = -hWidth; x <= hWidth; ++x)
    {
        float filterValue = expf( -(float)(x * x + y * y) / (2.f * sigma * sigma));
        int offset = (y + hWidth) * width + x + hWidth;

        h_filter[offset] = filterValue;
        filterSum += filterValue;
    }

    float normalizationFactor = 1.f / filterSum;

    for (int y = -hWidth; y <= hWidth; ++y)
    for (int x = -hWidth; x <= hWidth; ++x)
    {
        int offset = (y + hWidth) * width + x + hWidth;
        h_filter[offset] *= normalizationFactor;
    }

    return h_filter;
}


void preProcess(const std::string  &filename,
                      uchar4      **d_inputImageRGBA,
                      uchar4      **d_outputImageRGBA,
                      int2         *imageHW)
{
    cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty())
    {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    cv::Mat imageInputRGBA;
    cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

    const size_t imageSize = sizeof(uchar4) * image.rows * image.cols;

    checkCudaErrors( cudaMalloc(d_inputImageRGBA, imageSize) );
    checkCudaErrors( cudaMalloc(d_outputImageRGBA, imageSize) );

    checkCudaErrors( cudaMemset(*d_outputImageRGBA, 0, imageSize) );
    checkCudaErrors( cudaMemcpy(*d_inputImageRGBA,
                                imageInputRGBA.ptr<unsigned char>(0),
                                imageSize,
                                cudaMemcpyHostToDevice) );

    *imageHW = make_int2(image.cols, image.rows);
}


void postProcess(const std::string        &output_file,
                 const uchar4      *const  d_outputImageRGBA,
                 const int2        *const  imageHW)
{
    cv::Mat imageOutputRGBA;
    imageOutputRGBA.create(imageHW->y, imageHW->x, CV_8UC4);

    checkCudaErrors( cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0),
                                d_outputImageRGBA,
                                sizeof(uchar4) * imageHW->x * imageHW->y,
                                cudaMemcpyDeviceToHost) );

    cv::Mat imageOutputBGR;
    cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
    cv::imwrite(output_file.c_str(), imageOutputBGR);
}


int main(int argc, char **argv)
{
    checkCudaErrors(cudaFree(0));   // check if cuda context is initialzed

    if (argc != 3) 
    {
        std::cerr << "Usage: ./blur input_file output_file" << std::endl;
        exit(1);
    }

    std::string input_file  = std::string(argv[1]);
    std::string output_file = std::string(argv[2]);

    int filterWidth = 9;
    float *h_filter = generateBlurKernel(filterWidth, 2.);
    float *d_filter = d_allocFilter(h_filter, filterWidth);
    delete[] h_filter;


    uchar4 *d_inputImageRGBA,
           *d_outputImageRGBA;
    int2    imageHW;
    preProcess(input_file, &d_inputImageRGBA, &d_outputImageRGBA, &imageHW);


    GpuTimer timer;
    timer.start();
    gaussian_blur(d_inputImageRGBA, d_outputImageRGBA,
                  imageHW.y, imageHW.x,
                  d_filter, filterWidth);
    timer.stop();


    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    printf("%f ms\n", timer.getElapsed());

    postProcess(output_file, d_outputImageRGBA, &imageHW);

    cudaFree(d_filter);
    cudaFree(d_inputImageRGBA);
    cudaFree(d_outputImageRGBA);
    return 0;
}