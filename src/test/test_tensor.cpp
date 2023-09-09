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

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}