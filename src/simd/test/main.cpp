#include "../../../include/types/batch.h"
#include <iostream>
#include <cassert>

int main()
{
    tinyvsu::batch<int> a(1, 2, 3, 4);
    tinyvsu::batch<int> b(2, 3, 4, 5);
    tinyvsu::batch<int> c = a + b ;
    assert(c.get(0) == 3);
    assert(c.get(1) == 5);
    assert(c.get(2) == 7);
    assert(c.get(3) == 9);
    std::cout << "pass!" << std::endl;
    return 0;
}