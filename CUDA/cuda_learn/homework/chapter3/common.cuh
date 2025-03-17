#pragma once

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define CUDA_CHECK(call)   ErrorCheck(call,__FILE__,__LINE__)

//错误检查函数
cudaError_t ErrorCheck(cudaError_t error_code,const char* filename,const int lineNumber)
{
    if(error_code != cudaSuccess)
    {
        printf("CUDA error \r\ncode=%d,name=%s,description=%s\r\n at file=%s\r\n :line=%d \r\n",
        error_code,cudaGetErrorName(error_code),cudaGetErrorString(error_code),filename,lineNumber);
        return error_code;
    }
    return error_code;
}
//GPU设置函数
void setGPU(void)
{

    //检测计算机GPU数量
    int iDeviceCount = 0;
    //返回错误代码,cudaGetDeviceCount函数返回错误代码,并传入设备数量指针
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount),__FILE__,__LINE__);

    if(error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No GPU device found!\n");
        exit(-1);
    }
    else
    {
        printf("GPU device count: %d\n", iDeviceCount);
    }

    //设置执行，一块显卡默认id为0
    int iDevice = 0;
    error = cudaSetDevice(iDevice);
    if(error != cudaSuccess)
    {
        printf("Failed to set GPU device %d!\n", iDevice);
        exit(-1);
    }

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, iDevice));
    std::cout<<"运行GPU设备:"<<deviceProp.name<<std::endl;
    
}

double seconds(void)
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec*1.0e-6;
}

