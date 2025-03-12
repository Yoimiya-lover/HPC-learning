### 头文件
#include <omp.h>


执行方法
g++ -fopenmp test.cpp -o test

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
 一般与 #pragma omp parallel配合，表示多个线程可以同时执行不同任务，但是需要等待所有线程都执行完才继续执行
 #pragma omp section
 表示以下不同代码块会被分配不同的线程

 #pragma omp critical(A)
 表示A锁，不指定锁命默认critical使用同一把锁

omp_get_wtime()
返回lf时间

omp_get_thread_num()
获得线程id

 omp_set_nested(1)
 启动多层嵌套，内外多线程，嵌套并行

 omp_get_num_procs()
 获得处理器个数

 omp_get_max_threads()
 获得最大线程数

 #pragma omp for nowait
 该 `for` 循环结束后，不需要强制执行线程的隐式同步。意味着当一个线程完成了自己的工作后不需要等待，可以直接做另外的任务

 #pragma omp master` 
 指示只由主线程执行以下代码块。也就是说，即使有多个线程参与执行，只有主线程会执行该段代码，而其他线程将跳过这段代码。

#pragma omp parallel for schedule(dynamic)
 动态调度（dynamic）
任务分配方式：每个线程先分配一部分循环迭代任务，执行完后再领取新任务，直到所有任务完成。
适用场景：
每次循环的计算量不均匀（比如有些任务很重，有些任务很轻）。
适用于 不确定的计算任务，比如数据依赖或分支跳转的计算。


#pragma omp sections nowait
后的代码会等待所有section完成之后才执行，如果加入nowait，则每个线程完成section部分后会直接运行section之后的代码



