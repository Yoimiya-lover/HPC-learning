#include <stdio.h>
#include <cuda_runtime.h> 
#include <iostream>

//cudaGetDeviceProperties真能在主机端使用，不能写在核函数中
// __global__ void check_device()
// {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0); // 0 表示第一张 GPU
//     printf("Device Name: %s\n", prop.name);
//     printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
// }
int main()
{
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, 0); // 0 表示第一张 GPU
    printf("Device Name: %s\n",deviceProps.name);
    printf("Compute Capability: %d.%d\n", deviceProps.major, deviceProps.minor);
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;
    std::cout << "SM数量：" << deviceProps.multiProcessorCount << std::endl;
    std::cout << "L2缓存大小：" << deviceProps.l2CacheSize / (1024 * 1024) << "M" << std::endl;
    std::cout << "SM最大驻留线程数量：" << deviceProps.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "设备是否支持流优先级：" << deviceProps.streamPrioritiesSupported << std::endl;
    std::cout << "设备是否支持在L1缓存中缓存全局内存：" << deviceProps.globalL1CacheSupported << std::endl;
    std::cout << "设备是否支持在L1缓存中缓存本地内存：" << deviceProps.localL1CacheSupported << std::endl;
    std::cout << "一个SM可用的最大共享内存量：" << deviceProps.sharedMemPerMultiprocessor / 1024  << "KB" << std::endl;
    std::cout << "一个SM可用的32位最大寄存器数量：" << deviceProps.regsPerMultiprocessor / 1024 << "K" << std::endl;
    std::cout << "一个SM最大驻留线程块数量：" << deviceProps.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "GPU内存带宽：" << deviceProps.memoryBusWidth << std::endl;
    std::cout << "GPU内存频率：" << (float)deviceProps.memoryClockRate / (1024 * 1024) << "GHz" << std::endl;

}