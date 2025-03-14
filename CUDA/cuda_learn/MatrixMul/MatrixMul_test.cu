/*
*/

#include <iostream>
#include <stdio.h>
#include "MatrixClass.cuh"
#include "MatrixAlgorith.cuh"
#include "tool.cuh"
#include <ctime>

#define ROLL_NUM 200

enum GPU_METHOD {
    MULTIPLY,       // 默认值 0
    SHARE_MEMORY     // 1

};
enum CPU_METHOD {
    CPU_MULTIPLY,       // 默认值 0
    CPU_OMP_MULTIPLY     // 1

};

void test_method_host(MatrixMul::Matrix<float> &mat,CPU_METHOD method)
{
    std::clock_t c_start = std::clock();
    for(int i = 0; i < ROLL_NUM; i++)
    {
        if(method == CPU_MULTIPLY)
        {
            mat.multiply_cpu();
        }
        else if(method == CPU_OMP_MULTIPLY)
        {
            mat.multiply_cpu_omp();

        }
    }
    std::clock_t c_end = std::clock();
    float time_elaps_ms = 1000.0 * (c_end - c_start) / (ROLL_NUM * CLOCKS_PER_SEC);

    printf("Average execution time of %d roll launches = %f (ms)\n",ROLL_NUM,time_elaps_ms);

    mat.check_result();
    printf("计算完成");
}

void test_method_device(MatrixMul::Matrix<float> &mat,int M, int K, int N,GPU_METHOD method)

{
    mat.Initiate_C_zero();
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

        if(method == MULTIPLY)
        {
            mat.multiply(gridSize,blockSize);
        }
        else if(method == SHARE_MEMORY)
        {
            mat.multiply_share_memory(gridSize,blockSize);
        }

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
    printf("Average execution time of %d GPU kernel launches = %f (ms)\n",ROLL_NUM,t_ave);

    mat.cudaMem_Device_To_Host();
    mat.check_result();
   
}



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
        return -1;
    }
    setGPU();
    MatrixMul::Matrix<float> mat(M,K,N);
    mat.cudaMem_Host_To_Device();
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,(M + blockSize.y - 1) / blockSize.y);
    
    std::cout<<"--------开始测试CPU 初始算法--------"<<std::endl;
    test_method_host(mat,CPU_MULTIPLY);
    std::cout<<"--------开始测试CPU OMP算法--------"<<std::endl;
    test_method_host(mat,CPU_OMP_MULTIPLY);
    std::cout<<"--------开始测试GPU 算法--------"<<std::endl;
    test_method_device(mat, M, K, N,MULTIPLY);

    std::cout<<"--------开始测试GPU -共享内存算法--------"<<std::endl;
    test_method_device(mat, M, K, N,SHARE_MEMORY);


    
    mat.MatrixcudaDeviceReset();
    return 0;
}