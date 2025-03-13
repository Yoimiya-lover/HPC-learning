cmake_minimum_required(VERSION 3.20)

project(global_memory LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)# 指定cuda编译版本
set(CMAKE_CUDA_STANDARD_REQUIRED ON) #找不到指定版本编译器将发出错误
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDAToolkit_LIBRARY_DIRS}")

else()
    message(FATAL_ERROR "CUDA not found")
endif()


#添加可执行文件
add_executable(global_memory global_memory.cu)
target_include_directories(global_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../tool/)
target_include_directories(global_memory PRIVATE ${CUDA_INCLUDE_DIRS})

#链接库
target_link_libraries(global_memory PRIVATE ${CUDA_LIBRARIES})