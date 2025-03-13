
计时示例

cudaEvent_t start,stop;
ErrorCheck(cudaEventCreate(&start),__FILE__,__LINE__);
ErrorCheck(cudaEventCreate(&stop),__FILE__,__LINE__);
ErrorCheck(cudaEventRecord(start),__FILE__,__LINE__);
cudaEventQuery(start);

MatrixSum1D_GPU<<<grid,block>>>(fpDevice_A,fpDevice_B,fpDevice_C,iElemCount);

ErrorCheck(cudaEventRecord(stop),__FILE__,__LINE__);
ErrorCheck(cudaEventSynchronize(stop),__FILE__,__LINE__);
float elapse_time;
ErrorCheck(cudaEventElapsedTime(&elapse_time,start,stop),__FILE__,__LINE__);

if(repeat > 0)
{
    t_sum += elapse_time;
}
ErrorCheck(cudaEventDestroy(start),__FILE__,__LINE__);
ErrorCheck(cudaEventDestroy(stop),__FILE__,__LINE__);