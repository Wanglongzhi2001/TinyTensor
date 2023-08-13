#include "../src/vector/add.h"
#include <vector>
#include <iostream>
#include "./test_vectorAdd.h"
#include "../third_party/googletest/googletest/include/gtest/gtest.h"

using tinyvsu::speedUpMethod;

TEST(vectorAddGPUTest, doubleTest){
    std::vector<double> x{1.0,2.0,3.0};
    std::vector<double> y{4.0,5.0,6.0};
    std::vector<double> gt{5.0,7.0,9.0};
    std::vector<double> z = tinyvsu::vcAdd<double, speedUpMethod::GPU>(x, y);
    ASSERT_EQ(z, gt);
}

TEST(vectorAddGPUTest, floatTest){
    std::vector<float> x{1.0,2.0,3.0};
    std::vector<float> y{4.0,5.0,6.0};
    std::vector<float> gt{5.0,7.0,9.0};
    std::vector<float> z = tinyvsu::vcAdd<float, speedUpMethod::GPU>(x, y);
    ASSERT_EQ(z, gt);
}

TEST(vectorAddGPUTest, intTest){
    std::vector<int> x{1,2,3,4,5,6,7,8};
    std::vector<int> y{4,5,6,7,8,9,10,11};
    std::vector<int> gt{5,7,9,11,13,15,17,19};
    std::vector<int> z = tinyvsu::vcAdd<int, speedUpMethod::GPU>(x, y);
    ASSERT_EQ(z, gt);
}

TEST(vectorAddSIMDTest, doubleTest){
    std::vector<double> x{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    std::vector<double> y{4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0};
    std::vector<double> gt{5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0};
    std::vector<double> z = tinyvsu::vcAdd<double, speedUpMethod::SIMD>(x, y);
    ASSERT_EQ(z, gt);
}

TEST(vectorAddSIMDTest, floatTest){
    std::vector<float> x{1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    std::vector<float> y{4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0};
    std::vector<float> gt{5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0};
    std::vector<float> z = tinyvsu::vcAdd<float, speedUpMethod::SIMD>(x, y);
    ASSERT_EQ(z, gt);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}