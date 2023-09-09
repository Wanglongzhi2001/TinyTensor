#include "src/tensor.h"
#include <vector>
#include <iostream>
#include "googletest/include/gtest/gtest.h"

TEST(TensorTest, TestShape) {
    TT::Shape shape = {2, 3};
    auto tensor = new TT::Tensor(shape); 
    EXPECT_EQ(tensor->shape(0), 2);
    EXPECT_EQ(tensor->shape(1), 3);
}

TEST(TensorTest, TestDtype) {
    TT::Shape shape = {2, 3};
    auto tensor = new TT::Tensor(shape, TT::DType::Int32); 
    EXPECT_EQ(tensor->dtype(), TT::DType::Int32);
}

TEST(TensorTest, TestGet) {
    TT::Shape shape = {2, 3};
    std::vector<float> v = {1, 2, 3, 4, 5, 6};
    auto tensor = new TT::Tensor(v.data(), shape);
    EXPECT_EQ(static_cast<float*>(tensor->data_ptr())[0], 1);
    EXPECT_EQ(static_cast<float*>(tensor->data_ptr())[1], 2);
    EXPECT_EQ(static_cast<float*>(tensor->data_ptr())[2], 3);
    EXPECT_EQ(static_cast<float*>(tensor->data_ptr())[3], 4);
    EXPECT_EQ(static_cast<float*>(tensor->data_ptr())[4], 5);
    EXPECT_EQ(static_cast<float*>(tensor->data_ptr())[5], 6);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}