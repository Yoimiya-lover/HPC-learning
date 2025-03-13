/*
GPU缓存使用

*/

#include <stdio.h>
#include <iostream>
#include "common.cuh"

__global__ void  test_GPU_cache(void)
{

}


int main(int argc,char **argv)
{
    setGPU();

    cudaDeviceProp prop;
    if(prop.globalL1CacheSupported)
    {
        std::cout<<"GPU支持全局L1缓存"<<std::endl;
    }
    else
    {
        std::cout<<"GPU不支持全局L1缓存"<<std::endl;
    }

    std::cout<<"L2缓存大小"<<prop.l2CacheSize /(1024 * 1024)<<"M"<<std::endl;

    dim3 block(1);
    dim3 grid(1);
    test_GPU_cache<<<grid,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());//设备同步
    CUDA_CHECK(cudaDeviceReset());//设备重置

    return 0;

}