#include <stdio.h>
#include <cuda_runtime.h>  // 确保包含 CUDA 运行时库

__global__ void hello_from_gpu()
{
    //内建变量只在核函数有效
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int id = threadIdx.x + blockDim.x * blockIdx.x;
    //hello_from_gpu<<<2, 4>>>,2为grid_size,4为block_size
    //grid.Dim = 2    
    //block.Dim = 4
    printf("Hello World from the GPU block%d and thread %d global id:%d\n",bid,tid,id);
}

int main(void)
{
    printf("Hello World from the CPU\n");
    hello_from_gpu<<<2, 4>>>();  // 启动 4x4 个线程
    cudaError_t err = cudaGetLastError(); // 获取 CUDA 错误
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // 确保 GPU 任务执行完成

    return 0;
}
