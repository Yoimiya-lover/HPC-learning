#include <iostream>
#include <omp.h>
#include <unistd.h>

int parallel_wrong()
{
    int a[6] = {1,2,3,4,5,6};
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 6; i++)
    {
        sum += a[i];
    }
    return sum;
}
int parallel_right()
{
    int a[6] = {1,2,3,4,5,6};
    int sum = 0;
    #pragma omp parallel for 
    for (int i = 0; i < 6; i++)
    {
        sum += a[i];
    }
    return sum;
}

int main()
{
    std::cout<<"sum:"<<parallel_wrong()<<std::endl;
    std::cout<<"sum:"<<parallel_right()<<std::endl;
    return 0;
}