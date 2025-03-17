#include <iostream>
#include "common.cuh"

__global__ void warmingup(float *d_C,int size)
{
    for(int i = 0;i < size;i++)
    {
        d_C[i] = 1.0f;
    }
} 

__global__ void mathKernell1(float* c)
{
    int tid  = blockDim.x * blockIdx.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    if(tid % 2 == 0)
    {
        a = 100.f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernell2(float* c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float a ,b;
    a = b = 0.0f;
    if((tid/warpSize) % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    
    }
    c[tid] = a + b; 
}

int main(int argc,char** argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d: %s\n",dev,deviceProp.name);

    int size = 64;
    int blocksize = 64;
    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size = atoi(argv[2]);
    printf("Data size %d\n",size);

    dim3 block (blocksize,1);
    dim3 grid((size + block.x - 1)/block.x,1);
    printf("Execution Configure (block %d grid %d)\n",block.x,grid.x);

    float* d_C;
    int nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C,nBytes);

    size_t iStart,iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid,block>>>(d_C,size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("Warming <<<%4d %4d>>>elapsed %zu sec\n",grid.x,grid.y,iElaps);

    iStart = seconds();
    mathKernell1<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernell1<<<%4d %4d>>>elapsed %zu sec\n",grid.x,grid.y,iElaps);

    iStart = seconds();
    mathKernell2<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernell2<<<%4d %4d>>>elapsed %zu sec\n",grid.x,grid.y,iElaps);

    cudaFree(d_C);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}