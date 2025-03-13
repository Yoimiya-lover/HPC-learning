#pragma once

#include <stdio.h>
#include "../tool/common.cuh"

/*************************** 

line93：grid最多指定1024，这里声明了2048个，核函数调用报错

****************************/


__device__ float add(float fpDevice_A,float fpDevice_B)
{
    return fpDevice_A + fpDevice_B;
}

__global__ void MatrixSum1D_GPU(float *fpDevice_A,float *fpDevice_B,float *fpDevice_C,int iElemCount)
{
    int iThreadID = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(iThreadID >= iElemCount) return;
    fpDevice_C[iThreadID] = add(fpDevice_A[iThreadID],fpDevice_B[iThreadID]);
    
}

void MatrixSum1D_CPU(float *fpHost_A,float *fpHost_B,float *fpHost_C,int iElemCount)
{
    for(int i = 0;i < iElemCount;i++)
    {
        fpHost_C[i] = fpHost_A[i] + fpHost_B[i];
    }
}

void initialData(float *addr,int elemCount)
{
    for(int i = 0;i < elemCount;i++)
    {
        addr[i] = (float)(rand()%100)/10.0f;
    }
}


int main(void)
{
    //1.设置GPU设备
    setGPU();

    //2.分配主机内存和设备内存并初始化
    int iElemCount = 1024;                               //设置元素个数
    size_t stBytesCount = iElemCount * sizeof(float);   //设置字节数

    //分配主机内存初始化
    float *fpHost_A,*fpHost_B,*fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if(fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A,0,stBytesCount);
        memset(fpHost_B,0,stBytesCount);
        memset(fpHost_C,0,stBytesCount);
    }
    else
    {
        printf("malloc failed!\n");
        exit(-1);
    }

    // 分配设备内存并初始化
    float *fpDevice_A,*fpDevice_B,*fpDevice_C;
    ErrorCheck(cudaMalloc((float**)&fpDevice_A,stBytesCount),__FILE__,__LINE__);
    ErrorCheck(cudaMalloc((float**)&fpDevice_B,stBytesCount),__FILE__,__LINE__);
    ErrorCheck(cudaMalloc((float**)&fpDevice_C,stBytesCount),__FILE__,__LINE__);
    
    if(fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
    {
        ErrorCheck(cudaMemset(fpDevice_A,0,stBytesCount),__FILE__,__LINE__);
        ErrorCheck(cudaMemset(fpDevice_B,0,stBytesCount),__FILE__,__LINE__);
        ErrorCheck(cudaMemset(fpDevice_C,0,stBytesCount),__FILE__,__LINE__);
       
    }
    else
    {
        printf("cudaMalloc failed!\n");
        exit(-1);
    }
    //初始化主机中数据
    srand(666);//设置随机种子
    initialData(fpHost_A,iElemCount);
    initialData(fpHost_B,iElemCount);

    //数据从主机复制到设备
    ErrorCheck(cudaMemcpy(fpDevice_A,fpHost_A,stBytesCount,cudaMemcpyHostToDevice),__FILE__,__LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_B,fpHost_B,stBytesCount,cudaMemcpyHostToDevice),__FILE__,__LINE__);

    //调用核函数在设备进行计算
    dim3 block(2048);//设置线程块大小,最大为1024
    dim3 grid((iElemCount + block.x - 1 )/ block.x);//设置线程块数量,向上取整

    //调用该行会报错
    MatrixSum1D_GPU<<<grid,block>>>(fpDevice_A,fpDevice_B,fpDevice_C,iElemCount);
    ErrorCheck(cudaGetLastError(),__FILE__,__LINE__);
    ErrorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__);

    //计算数据从设备传回主机
    ErrorCheck(cudaMemcpy(fpHost_C,fpDevice_C,stBytesCount,cudaMemcpyDeviceToHost),__FILE__,__LINE__);

    for(int i = 0;i < 10;i++)
    {
        printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tmatrix_C:%.2f\n",i+1,fpHost_A[i],fpHost_B[i],fpHost_C[i]);

    }

    //释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    ErrorCheck(cudaFree(fpDevice_A),__FILE__,__LINE__);
    ErrorCheck(cudaFree(fpDevice_B),__FILE__,__LINE__);
    ErrorCheck(cudaFree(fpDevice_C),__FILE__,__LINE__);
    
    //重置
    ErrorCheck(cudaDeviceReset(),__FILE__,__LINE__);
    return 0;
}