### 头文件
#include <omp.h>

### 该示例的几个API接口
 #pragma omp parallel num_threads(4) default(none) shared(data,std::cout)
 启动四个线程，default(none)要求所有变量都要显示指定
 data变量所有线程均可访问
 std::cout实际上是一个全局变量，所以需要显示指定

 #pragma omp single
 表示只需要一个线程执行以下语句

 #pragma omp barrier
 表示所有线程都执行完该命令后，才继续执行



