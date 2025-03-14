#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include "MatrixAlgorith.cuh"
#include "tool.cuh"

namespace MatrixMul
{
    template <typename T>
    __global__ void matrixMulKernel(T* A, T* B, T* C,const int M, const int K, const int N) 
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    template <typename T>
    __global__ void matrixMulkernel_v2(T* A, T* B, T* C,const int M, const int K, const int N)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        int tx = threadIdx.x;
        int ty = threadIdx.y;

        __shared__ T SA[32][32];
        __shared__ T SB[32][32];
        int width =( K + blockDim.x -1 )/ blockDim.x;

        T Csum = 0;
        for (int ph = 0; ph < width; ph++) 
        {
            // 确保索引合法，避免越界访问
            if (row < M && (ph * blockDim.x + tx) < K) {
                SA[ty][tx] = A[row * K + (ph * blockDim.x + tx)];
            } else {
                SA[ty][tx] = 0.0f;
            }
    
            if ((ph * blockDim.x + ty) < K && col < N) {
                SB[ty][tx] = B[(ph * blockDim.x + ty) * N + col];
            } else {
                SB[ty][tx] = 0.0f;
            }
    
            __syncthreads();
    
            // 计算 C 的局部和
            for(int k = 0; k < blockDim.x; k++) {
                Csum += SA[ty][k] * SB[k][tx];
            }
    
            __syncthreads();
        }
    
        // 将计算结果写入全局内存
        if (row < M && col < N) {
            C[row * N + col] = Csum;
        }
    }

    template<typename T>
    class Matrix{
        private:
            T* _data_A_Host;
            T* _data_B_Host;
            T* _data_C_Host;

            T* _data_A_Device;
            T* _data_B_Device;
            T* _data_C_Device;

            const int _M;
            const int _N;
            const int _K;

            const int _A_size = _M * _K;
            const int _B_size = _N * _K;
            const int _C_size = _M * _N;
        public:
            Matrix(const int M, const int K,const int N):_M(M),_K(K),_N(N)
            {
                _data_A_Host = new T[_M*_K];
                _data_B_Host = new T[_K*_N];
                _data_C_Host = new T[_M*_N];
                Initiate_Host(_data_A_Host,M,K);
                Initiate_Host(_data_B_Host,K,N);
                Initiate_Host(_data_C_Host,M,N,0);
                std::cout << "主机端矩阵创建完成，并随机赋值" << std::endl;

                Initiate_Device(_data_A_Device,_A_size);
                Initiate_Device(_data_B_Device,_B_size);
                Initiate_Device(_data_C_Device,_C_size,0);

                std::cout << "设备端矩阵创建完成" << std::endl;
            }

            void Initiate_Host(T* _data,int row,int col)
            {
                srand(time(NULL));//随机种子
                for(int i = 0; i < row * col; i++)
                {
                    _data[i] = rand() % 10;
                }
                //std::cout << "创建矩阵并初始化成功!!!" << std::endl;
            }

            void Initiate_Host(T* _data, int row,int col,int value)
            {
                for(int i = 0; i < row * col; i++)
                {
                    _data[i] = value;
                }
                //std::cout << "主机端矩阵初始化成功!!!" << std::endl;
            }

            void Initiate_Device(T*& _data, int size,int value)
            {
                CUDA_CHECK(cudaMalloc((void**)&_data,size * sizeof(T)));
                CUDA_CHECK(cudaMemset(_data, value, size * sizeof(T))); 
            }
            void Initiate_Device(T*& _data, int size)
            {
                CUDA_CHECK(cudaMalloc((void**)&_data,size * sizeof(T)));
            }


            void Initiate_C_zero()
            {
                Initiate_Device(_data_C_Device,_C_size,0);
            }

            void cudaMem_Host_To_Device()
            {
                CUDA_CHECK(cudaMemcpy(_data_A_Device, _data_A_Host, sizeof(T) * _A_size, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(_data_B_Device, _data_B_Host, sizeof(T) * _B_size, cudaMemcpyHostToDevice));
                //std::cout<<"_data_A_Device[0][0] = "<<_data_A_Device[0]<<std::endl;
                std::cout<<"_data_A_Host[0][0] = "<<_data_A_Host[0]<<std::endl;
                std::cout<<"数据已从主机端传输到设备端"<<std::endl;
            }

            

            void multiply(dim3 grid,dim3 block)
            {
                matrixMulKernel<float><<<grid,block>>>(_data_A_Device,_data_B_Device,_data_C_Device,_M,_K,_N);
                this->MatrixcudaDeviceSynchronize();
            }

            void multiply_share_memory(dim3 grid,dim3 block)
            {
                matrixMulkernel_v2<float><<<grid,block>>>(_data_A_Device,_data_B_Device,_data_C_Device,_M,_K,_N);
                this->MatrixcudaDeviceSynchronize();
            }

            void multiply_cpu()
            {
                MatrixAlgorith::MatrixMulOrigin(_data_A_Host, _data_B_Host, _data_C_Host, _M, _K, _N);
            }

            void multiply_cpu_omp()
            {
                MatrixAlgorith::MatrixMulOmp(_data_A_Host, _data_B_Host, _data_C_Host, _M, _K, _N);
            }

            void cudaMem_Device_To_Host()
            {
                CUDA_CHECK(cudaMemcpy(_data_C_Host, _data_C_Device, sizeof(T) * _C_size, cudaMemcpyDeviceToHost));
                std::cout<<"数据已从设备端传输到主机端"<<std::endl;
            }

            void check_result()
            {
                T* result = new T[_C_size];
                MatrixAlgorith::MatrixMulOrigin(_data_A_Host, _data_B_Host, result, _M, _K, _N);
                //for(int i = 0; i < _C_size; i++)
                for(int i = 0; i < _M; i++) 
                {
                    for(int j = 0; j < _N; j++) {
                        if(result[i*_N+j] != _data_C_Host[i*_N+j])
                        {
                            std::cout<<"Error!!"<<"C["<<i<<"]["<<j<<"]  is wrong!"<<std::endl;
                            std::cout<<"c_host["<<i<<"]["<<j<<"] = "<<_data_C_Host[i*_N+j]<<"  c_result["<<i<<"]["<<j<<"] = "<<result[i*_N+j]<<std::endl;
                            return;
                        }
                    }
                }
                std::cout<<"计算无误"<<std::endl;
                delete[] result;

            }

            void MatrixcudaDeviceSynchronize()
            {
                cudaDeviceSynchronize();
                //std::cout<<"GPU与CPU同步完成"<<std::endl;
            }
            
            void MatrixcudaDeviceReset()
            {
                cudaDeviceReset();
                std::cout<<"重置完成"<<std::endl;
            }

            ~Matrix()
            {
                delete [] _data_A_Host;
                delete [] _data_B_Host;
                delete [] _data_C_Host;
                std::cout<<"删除主机端矩阵成功"<<std::endl;
                cudaFree(_data_A_Device);
                cudaFree(_data_B_Device);
                cudaFree(_data_C_Device);
                std::cout<<"删除设备端矩阵成功"<<std::endl;

            }


            


    };
}