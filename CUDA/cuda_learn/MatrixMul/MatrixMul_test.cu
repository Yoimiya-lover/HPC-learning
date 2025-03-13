/*
*/

#include <iostream>
#include <stdio.h>
#include "MatrixClass.cuh"
#include "MatrixAlgorith.cuh"
#include "tool.cuh"

#define ROLL_NUM 200

int main(int argc,const char* argv[])
{
    if(argc < 4) {
        std::cout << "参数设置错误, 至少需要3个参数!!!" << std::endl;
        return -1;
    }
    const int M = atoi(argv[1]);
    const int K = atoi(argv[2]);
    const int N = atoi(argv[3]);
    if(M <=0 || K <= 0 || N <= 0 ) {
        std::cout << "参数设置错误！！！" << "正确参数为: M > 0; K > 0; N > 0; Thread_num >= 0 !!!" << std::endl;
    }
    setGPU();
    MatrixMul::Matrix<float> mat(M,K,N);
    mat.cudaMem_Host_To_Device();
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,(M + blockSize.y - 1) / blockSize.y);
    float t_sum = 0;
    for(int i = 0; i < ROLL_NUM; i++)
    {
        cudaEvent_t start,stop;
        ErrorCheck(cudaEventCreate(&start),__FILE__,__LINE__);
        ErrorCheck(cudaEventCreate(&stop),__FILE__,__LINE__);
        ErrorCheck(cudaEventRecord(start),__FILE__,__LINE__);
        cudaEventQuery(start);

        mat.multiply(gridSize, blockSize);

        ErrorCheck(cudaEventRecord(stop),__FILE__,__LINE__);
        ErrorCheck(cudaEventSynchronize(stop),__FILE__,__LINE__);
        float elapse_time;
        ErrorCheck(cudaEventElapsedTime(&elapse_time,start,stop),__FILE__,__LINE__);

        if(i > 0)
        {
            t_sum += elapse_time;
        }
        ErrorCheck(cudaEventDestroy(start),__FILE__,__LINE__);
        ErrorCheck(cudaEventDestroy(stop),__FILE__,__LINE__);
        
    }
    const float t_ave = t_sum / ROLL_NUM;
    printf("Average execution_v1 time of %d GPU kernel launches = %f (ms)\n",ROLL_NUM,t_ave);

    mat.cudaMem_Device_To_Host();
    mat.MatrixcudaDeviceReset();

    //mat.check_result();
    printf("计算完成");

    MatrixMul::Matrix<float> mat2(M,K,N);
    mat2.cudaMem_Host_To_Device();
    dim3 blockSize2(32, 32);
    dim3 gridSize2((N + blockSize2.x - 1) / blockSize2.x,(M + blockSize2.y - 1) / blockSize2.y);
    float t_sum_v2 = 0;
    for(int i = 0; i < ROLL_NUM; i++)
    {
        cudaEvent_t start,stop;
        ErrorCheck(cudaEventCreate(&start),__FILE__,__LINE__);
        ErrorCheck(cudaEventCreate(&stop),__FILE__,__LINE__);
        ErrorCheck(cudaEventRecord(start),__FILE__,__LINE__);
        cudaEventQuery(start);

        mat2.multiply_v2(gridSize2, blockSize2);

        ErrorCheck(cudaEventRecord(stop),__FILE__,__LINE__);
        ErrorCheck(cudaEventSynchronize(stop),__FILE__,__LINE__);
        float elapse_time;
        ErrorCheck(cudaEventElapsedTime(&elapse_time,start,stop),__FILE__,__LINE__);

        if(i > 0)
        {
            t_sum_v2 += elapse_time;
        }
        ErrorCheck(cudaEventDestroy(start),__FILE__,__LINE__);
        ErrorCheck(cudaEventDestroy(stop),__FILE__,__LINE__);
        
    }
    const float t_ave_v2 = t_sum_v2 / ROLL_NUM;
    printf("Average execution_v2 time of %d GPU kernel launches = %f (ms)\n",ROLL_NUM,t_ave_v2);

    mat2.cudaMem_Device_To_Host();
    mat2.check_result();
    mat2.MatrixcudaDeviceReset();

    return 0;


}