/*
 *
 *    动态共享内存使用
 *    核函数外，使用extern修饰符
 *    定义数组时extern __shared__ float s_array[]
 *    调用核函数时，test_dynamic_shared_memory<<<grid,block,32>>>(d_A,nElems);要指定动态内存大小
*/



#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.cuh"

extern __shared__ float s_array[];//动态共享内存

__global__ void test_dynamic_shared_memory(float* d_A,int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
   
    if(n < N)
    {
        s_array[tid] = d_A[n];
    }
    __syncthreads();//线程同步

    if(tid == 0)//同一网格共享相同的共享内存，不同网格共享内存不一致
    {
        for(int i = 0;i < 32;i++)
        {
            printf("kernel_1 : %f,blockIdx: %d\n",s_array[i],bid);
        }
    }
}

int main(int argc,char **argv)
{
    setGPU();

    int nElems = 64;
    int nBytes = nElems * sizeof(float);

    float *h_A = (float *)malloc(nBytes);
    for(int i = 0;i < nElems;i++)
    {
        h_A[i] = i;
    }

    float* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,nBytes));
    CUDA_CHECK(cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(2);
    test_dynamic_shared_memory<<<grid,block,32 * sizeof(float)>>>(d_A,nElems);

    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    CUDA_CHECK(cudaDeviceReset());

}