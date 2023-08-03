#include "../src/vector/add.h"
#include <vector>
#include <iostream>
#include "./test_vectorAdd.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"

TEST(vectorAddTest, vectorAddGPU){
    std::vector<double> x{1.0,2.0,3.0};
    std::vector<double> y{4.0,5.0,6.0};
    std::vector<double> gt{5.0,7.0,9.0};
    std::vector<double> z = vc::vcAdd<double, true>(x, y);
    ASSERT_EQ(z, gt);
}

TEST(vectorAddTest, vectorAddSIMD){
    std::vector<double> x{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    std::vector<double> y{4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0};
    std::vector<double> gt{5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0};
    std::vector<double> z = vc::vcAdd<double, false>(x, y);
    ASSERT_EQ(z, gt);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}