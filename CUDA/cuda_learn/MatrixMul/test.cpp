#include <iostream>
#include <omp.h>
#include <ctime>


using ElemType = float;
    /* 最原始的三层for循环计算AxB=C, M->N->K */
void MatrixMulOrigin(const ElemType* A, const ElemType *B, ElemType *C, const int& M, const int& K, const int& N)
{
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C[i*N+j] = 0;
            for(int k = 0; k < K; k++)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
        }
    }
}

void MatrixMulOmp(const ElemType* A, const ElemType *B, ElemType *C, const int& M, const int& K, const int& N)
{
    #pragma omp parallel for schedule(dynamic, 16)
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            C[i*N+j] = 0;
            for(int k = 0; k < K; k++)
                C[i*N+j] += A[i*K+k] * B[k*N+j];
        }
    }
    
}

int main()
{
    int M = 100,K = 100,N = 100;
    ElemType *A = new ElemType[M*K];
    ElemType *B = new ElemType[K*N];
    ElemType *C = new ElemType[M*N];
    srand(time(NULL));//随机种子
    for(int i = 0;i < M*K;i++)
    {
        A[i] = rand() % 10;
    }
    srand(time(NULL));//随机种子
    for(int i = 0;i < K*N;i++)
    {
        B[i] = rand() % 10;
    }
    MatrixMulOmp(A, B, C, M,  K, N);
    ElemType *C_result = new ElemType[M*N];
    MatrixMulOrigin(A, B, C_result, M,  K, N);
    for(int i = 0;i < M * N;i++)
    {
        if(C[i] != C_result[i])
        {
            std::cout<<"errorC["<<i<<"]"<<std::endl;
            return -1;
        }
        
    }
    std::cout<<"success"<<std::endl;



}    