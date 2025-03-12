#include <iostream>
#include <omp.h>
#include <unistd.h>
void origin()
{
    float start = omp_get_wtime();  
    #pragma parallel for num_threads(4) defalut(none) shared(start)
    {
        
        for(int i = 0;i < 4;i++)
        {
            sleep(i);
            printf("tid : %d time cost:%lf\n",omp_get_thread_num(),omp_get_wtime() - start);
        }
       

    }
    float end = omp_get_wtime();
    printf("origin time cost:%f\n",end - start);

}
void nowait()
{
    float start = omp_get_wtime();  
    #pragma parallel for num_threads(4) defalut(none) shared(start)
    {
        #pragma parallel for nowait
        {
            for(int i = 0;i < 4;i++)
            {
                sleep(i);
                printf("tid : %d time cost:%lf\n",omp_get_thread_num(),omp_get_wtime() - start);
            }
        }
       

    }
    float end = omp_get_wtime();
    printf("origin time cost:%f\n",end - start);

}

int main()
{
    origin();
    nowait();
    return 0;

}