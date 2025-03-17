#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc,char **argv)
{
    printf("%s Starting...\n",argv[0]);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if(error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n->%s",
                        (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    
    if(deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n",deviceCount);
    }

    int dev,driverVersion = 0,runtimeVersion = 0;

    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d: \"%s\"\n",dev,deviceProp.name);
    printf("最大线程数/块：%d\n",deviceProp.maxThreadsPerBlock);
    printf("最大块维度：%d,%d,%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("最大网格维度：%d,%d,%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    return 0;
    
    
}