cmake_minimum_required(VERSION 3.20)

project(cudaMatrixMul LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)# 指定cuda编译版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON) #找不到指定版本编译器将发出错误
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-Xptxas -dlcm=ca 强制 L1 缓存行为
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86 -Xptxas -dlcm=ca")

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDAToolkit_LIBRARY_DIRS}")

else()
    message(FATAL_ERROR "CUDA not found")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#添加可执行文件
add_executable(cudaMatrixMul MatrixMul_test.cu)
target_include_directories(cudaMatrixMul PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../tool/)
target_include_directories(cudaMatrixMul PRIVATE ${CUDA_INCLUDE_DIRS})

#链接库
target_link_libraries(cudaMatrixMul PRIVATE ${CUDA_LIBRARIES})