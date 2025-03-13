#pragma once
#include <stdio.h>
#include "../tool/common.cuh"

/**********************************
 * cudaMalloc,cudaMemcpy,cudaDeviceReset,cudaFree都是返回cudaError_t
 * __FILE__是预处理宏，表示当前文件名
 * __LINE__是预处理宏，表示当前行号
 * 该文件为了测试cuda_error的错误检查函数
 * 
************************************/
int main(void)
{
    int iElemCount = 100;
    float* fpHost_A = (float *)malloc(sizeof(float)*iElemCount);
    memset(fpHost_A,0,sizeof(float)*iElemCount);

    float* fpDevice_A ;
    cudaError_t error = ErrorCheck(cudaMalloc((float**)&fpDevice_A,sizeof(float)*iElemCount),__FILE__,__LINE__);
    cudaMemset(fpDevice_A,0,sizeof(float)*iElemCount);
    ErrorCheck(cudaMemcpy(fpDevice_A,fpHost_A,sizeof(float)*iElemCount,cudaMemcpyDeviceToHost),__FILE__,__LINE__);//这里有问题，cudaMemcpyDeviceToHost设备数据传输给主机

    free(fpHost_A);
    ErrorCheck(cudaFree(fpDevice_A),__FILE__,__LINE__);
    ErrorCheck(cudaDeviceReset(),__FILE__,__LINE__);
    return 0;
}
