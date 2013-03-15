#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>
#include <cuda_runtime.h>       // Error Handling, Event Management

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)


template<typename T>
void check(T err, const char* const func, const char* const file, const int line) 
{
  if (err != cudaSuccess) 
  {
    std::cerr << "cuda error at " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


struct GpuTimer
{
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    GpuTimer()
    {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent, 0);
    }

    void stop()
    {
        cudaEventRecord(stopEvent, 0);
    }

    float getElapsed()
    {
        float elapsed;
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        return elapsed;
    }
};

#endif  /* UTILS_H__ */
