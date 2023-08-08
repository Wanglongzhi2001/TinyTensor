#include "../types/batch.h"

int main()
{
    vsu::batch<int> a(1, 2, 3, 4);
    vsu::batch<int> b(1, 2, 3, 4);
    vsu::batch<int> c = a + b;
    return 0;
}