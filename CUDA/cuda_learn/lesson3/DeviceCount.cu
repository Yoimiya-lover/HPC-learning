#pragma once

#include <stdio.h>


void setGPU(void)
{

    //检测计算机GPU数量
    int iDeviceCount = 0;
    //返回错误代码
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);
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
    
}

