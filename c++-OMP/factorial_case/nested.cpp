#include <iostream>
#include <omp.h>
#include <unistd.h>

int main(void)
{
    omp_set_nested(1);
    #pragma omp parallel num_threads(2) default(none) shared(std::cout)
    {
        int parents_id = omp_get_thread_num();
        printf("parents id: %d\n",parents_id);
        sleep(2);
        #pragma omp barrier
        #pragma omp parallel num_threads(2) default(none) shared(parents_id)
        {
            sleep(parents_id + 1);
            printf("parents_id : %d child id: %d\n",parents_id,omp_get_thread_num());
            #pragma omp barrier
            printf("after barrier : parent_id = %d tid = %d\n", parents_id, omp_get_thread_num());
        }

    }
    return 0;
}