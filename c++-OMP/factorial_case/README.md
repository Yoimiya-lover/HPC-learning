### 头文件
#include <omp.h>

### 该示例的几个API接口
 #pragma omp parallel num_threads(4) default(none) shared(data,std::cout)
 启动四个线程，default(none)要求所有变量都要显示指定
 data变量所有线程均可访问
 std::cout实际上是一个全局变量，所以需要显示指定，如果在其他情况下，需要用#pragma omp critical来加锁，该变量只能由一个线程访问

 #pragma omp single
 表示只需要一个线程执行以下语句

 #pragma omp barrier
 表示所有线程都执行完该命令后，才继续执行

 #pragma omp sections
 表示多个线程可以同时执行，但是需要等待所有线程都执行完才继续执行
 #pragma omp section
 表示以下不同代码块会被分配不同的线程

 #pragma omp critical(A)
 表示A锁，不指定锁命默认critical使用同一把锁

omp_get_wtime()
返回lf时间

omp_get_thread_num()
获得线程id

