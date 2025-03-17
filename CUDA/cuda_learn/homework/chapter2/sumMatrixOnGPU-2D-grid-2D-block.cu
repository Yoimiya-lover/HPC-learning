#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <ctime>

void checkResult(const float *hostRef,const float *gpuRef,const int N)
{
    double eplision = 1.0E-6;
    int match = 1;
    for(int i = 0 ;i < N;i++)
    {
        if(i == N - 1) 
        {
            printf("gpu[N - 1] = %lf\n",gpuRef[N - 1]);
            printf("host[N - 1] = %lf\n",hostRef[N - 1]);
        }
        
        if(fabs(hostRef[i] - gpuRef[i]) > eplision)
        {
            
            match = 0;
            printf("Index %d: %f %f\n",i,hostRef[i],gpuRef[i]);
            printf("ERROR!!\n");
            break;
        }
    }
    if(match)
    {
        printf("Check Success!\n");
    
    }
    return;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec*1.e-6;
}

void initialData(float*data,int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0;i < size;i++)
    {
        data[i] = (float)(rand() & 0xFF)/10.0f;
    }

}

void sumMatrixOnCPU(float* A,float* B,float* C,const int nx,const int ny)
{
    for(int i = 0;i < nx;i++)
    {
        for(int j = 0;j < ny;j++)
        {
            int idx = i*ny+j;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void sumMatrixOnGPU(float* A,float* B,float* C,const int nx,const int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    int idx = ix*ny+iy;
    //printf("stride = %d\n",stride);
    if(idx < nx*ny)
    {
        C[idx] = A[idx] + B[idx];
        //单个线程计算两个元素
        // if(idx + stride < nx*ny)
        // {
        //     //printf("stride\n");
        //     C[idx + stride] = A[idx + stride] + B[idx + stride];
        // }   

        //单个线程计算三个元素
        // if(idx + stride * 2 < nx*ny)
        // {
        //     int id = idx + stride * 2;
        //     //printf("stride\n");
        //     C[id] = A[id] + B[id];
        // }
     
        // if(idx + stride * 2 < nx*ny)
        // {
        //     C[idx + stride] = A[idx + stride] + B[idx + stride];
        // }
    }
}

int main(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device:%d, %s\n", dev, deviceProp.name);

    //set up data for matrix
    int nx = 1<<10;
    int ny = 1<<10;
    int nxy = nx*ny;
    int nBytes = nxy*(sizeof(float));
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    //malloc host mem
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    //initialize data at host side
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuSecond() - iStart;
    printf("initial host matrix:%f\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnCPU(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("add matrix at host side:%f\n", iElaps);

    //malloc device global mem
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    //invoke cuda kernel
    int dimx = 8;
    int dimy = 8;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add matrix at device %f\n", iElaps);
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

    //copy kernel result to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    //check device result
    checkResult(hostRef, gpuRef, nxy);
    printf("gpuRef[nxy - 10] = %lf",gpuRef[nxy - 10]);
    printf("hostRef[nxy - 10] = %lf",hostRef[nxy -10]);
    
    //transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    //free device global memory
    free(hostRef);
    free(gpuRef);
    free(h_A);
    free(h_B);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    //reset device
    cudaDeviceReset();

    return 0;
    

}


