#include <iostream>
#include <omp.h>

int factorial(int n)
{
    int result = 1;
    if(n == 1) return 1;
    result = factorial(n - 1) * n;
    return result;
}

int main(void)
{
    int data[4] = {0};
    #pragma omp parallel num_threads(4) default(none) shared(data,std::cout)
    {
        int id = omp_get_thread_num();
        data[id] = factorial(id + 1);
        #pragma omp barrier
        
        #pragma omp single
        {
            long sum = 0;
            for(int i = 0;i < 4;i++)
            {
                sum += data[i];
            }
            
            std::cout<<"4的阶乘和为"<<sum<<std::endl;
        }
       
        }
        
}
