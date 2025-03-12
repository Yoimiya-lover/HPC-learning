#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <sstream>

/**************
 * 
 * 不建议使用std::cout来打印输出，会出现粘连
 * std::cout 是线程共享的，多个线程可能同时写入 
 * std::cout，导致输出的字符串被打断或交错。
 * 运算符 << 是多个独立操作，在 std::cout 进行一次完整的打印之前，其他线程可能插入输出，造成混杂。
 ***************/
int main()
{
    #pragma omp parallel sections num_threads(3) //default(none) shared(std::cout)
    {
        #pragma omp section
        {
            #pragma omp critical(A)
            {
                printf("Section A  ID:%d  time stamp:%lf \n",omp_get_thread_num(),omp_get_wtime());
                //std::cout << "Section A  ID:" <<omp_get_thread_num()<<"time stamp:"<<omp_get_wtime()<< std::endl;
                sleep(2);
            }
        }
        #pragma omp section
        {
            #pragma omp critical(B)
            {
                printf("Section B  ID:%d  time stamp:%lf \n",omp_get_thread_num(),omp_get_wtime());
                //std::cout<<"Section B  ID:"<<omp_get_thread_num()<<"time stamp:"<<omp_get_wtime()<< std::endl;
                sleep(2);
            }
        }
        #pragma omp section
        {
            #pragma omp critical(C)
            {
                printf("Section C  ID:%d  time stamp:%lf \n",omp_get_thread_num(),omp_get_wtime());
                //std::cout<<"Section C  ID:"<<omp_get_thread_num()<<"time stamp:"<<omp_get_wtime()<< std::endl;
                sleep(2);
            }

        }
    }
    return 0;
}