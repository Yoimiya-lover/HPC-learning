#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../tool/common.cuh"

/*静态全局变量使用*/

//静态全局变量必须定义在函数外
__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void test_global_memory(void)
{
    //定义在核函数外，但是核函数可以使用全局变量
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_y[0] = %d,d_y[1] = %d\n",d_y[0],d_y[1]);
}

int main(void)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps,devID));
    std::cout<<"运行GPU设备:"<<deviceProps.name<<std::endl;

    int h_y[2] = {10,20};
    //主机端传输到常量内存
    CUDA_CHECK(cudaMemcpyToSymbol(d_y,h_y,sizeof(int) * 2));

    dim3 block(1);
    dim3 grid(1);
    test_global_memory<<<grid,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    //常量内存传回主机端
    CUDA_CHECK(cudaMemcpyFromSymbol(h_y,d_y,sizeof(int) * 2));
    printf("h_y[0] = %d,h_y[1] = %d\n",h_y[0],h_y[1]);

    CUDA_CHECK(cudaDeviceReset());

    return 0;


}