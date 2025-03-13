errorCheckFuntion:检查函数的错误，device函数一般都返回cudaError_t
一般使用：
errorCheckFuntion(cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

errorCheckKernel:检查核函数的错误
由于核函数没有返回值，必须使用，cudaGetLastError()检查错误，例如：
MatrixSum1D_GPU<<<grid,block>>>(fpDevice_A,fpDevice_B,fpDevice_C,iElemCount);
ErrorCheck(cudaGetLastError(),__FILE__,__LINE__);
ErrorCheck(cudaDeviceSynchronize(),__FILE__,__LINE__);
