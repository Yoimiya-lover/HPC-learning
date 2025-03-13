#include <stdio.h>
//#include <cuda_runtime.h>  // 确保包含 CUDA 运行时库

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU\n");
}

int main(void)
{
    hello_from_gpu<<<4, 4>>>();  // 启动 4x4 个线程
    cudaError_t err = cudaGetLastError(); // 获取 CUDA 错误
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // 确保 GPU 任务执行完成

    return 0;
}
