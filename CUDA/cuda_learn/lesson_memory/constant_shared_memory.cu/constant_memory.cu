/*
 *
 *    常量内存使用
 *
 *
*/



#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void test_constant_memory(void)
{
    printf("Constant c_data = %.2f\n",c_data);
}

int main(int argc,char **argv)
{
    setGPU();

    float h_data = 8.8f;
   
    CUDA_CHECK(cudaMemcpyToSymbol(c_data,&h_data,sizeof(float)));

    dim3 block(1);
    dim3 grid(1);
    test_constant_memory<<<grid,block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_data,c_data2,sizeof(float)));
    printf("Constant c_data2 = %.2f\n",h_data);
    
    CUDA_CHECK(cudaDeviceReset());

}