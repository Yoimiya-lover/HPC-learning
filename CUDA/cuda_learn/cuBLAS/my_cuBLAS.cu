#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define M 4
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/* 把坐标转换为列优先坐标 */

static __inline__ void modify(cublasHandle_t handle,float *m,int ldm,int n,int p,int q,float alpha,float beta)
{
    cublasSscal(handle,n-q,&alpha,&m[IDX2C(p,q,ldm)],ldm);
    cublasSscal(handle,ldm-p,&beta,&m[IDX2C(p,q,ldm),1]);
}

int main(void)
{
    cudaError_t cudaStatus;
    cublasStatus_t status;
    cublasHandle_t handle;
    int i,j;
    float* devPtrA;
    float* a = 0;
    a = (float *)malloc (M * N * sizeof(*a));
    if(!a)
    {
        printf("host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    for(j = 0;j < N;j++)
    {
        for(i = 0;j < M;j++)
        {
            a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA,M*N*sizeof(*a));
    if(cudaStat != cudaSuccess)
    {
        printf("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublas initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix(M,N,sizeof(*a),a,M,devPtrA,M);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("data download failed\n");
        cudaFree(devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle,devPtrA,M,N,1,2,16.0f,12.0f);
    stat = cublasGetMatrix(M,N,sizeof(*a),devPtrA,M,a,M);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("data upload failed");
        cudaFree(devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for(int j = 0;j < N;j++)
    {
        for(int i = 0;i < M;i++)
        {
            printf("%7.0f",a[IDX2C(i,j,M)]);
        }
        printf("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}

