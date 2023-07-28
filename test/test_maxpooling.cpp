//
// Created by fss on 23-1-1.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "operator/Operator.h"
#include "layer/Layer.h"
#include "operator/MaxPoolingOperator.h"
#include "LayerRegister.h"

TEST(test_layer, forward_maxpooling1) {
  using namespace rq;
  uint32_t stride_h = 1;
  uint32_t stride_w = 3;
  uint32_t padding_h = 0;
  uint32_t padding_w = 0;
  uint32_t pooling_h = 3;
  uint32_t pooling_w = 3;

  std::shared_ptr<Operator> max_op = std::make_shared<MaxPoolingOperator>(pooling_h, pooling_w, stride_h, stride_w, padding_h, padding_w);
  std::shared_ptr<Layer<float>> max_layer = LayerRegister<float>::creatorLayer(max_op);
  CHECK(max_layer != nullptr);

  arma::fmat input_data = "71 22 63 94  65 16 75 58  9  11;"
                          "12 13 99 31 -31 55 99 857 12 511;"
                          "52 15 19 81 -61 15 49 67  12 41;"
                          "41 41 61 21 -15 15 10 13  51 55;";
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_data.n_rows, input_data.n_cols, 1);
  input->at(0) = input_data;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);

  max_layer->forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  const auto &output = outputs.at(0);

  ASSERT_EQ(output->rows(), 2);
  ASSERT_EQ(output->cols(), 3);

  ASSERT_EQ(output->at(0, 0, 0), 99);
  ASSERT_EQ(output->at(0, 0, 1), 94);
  ASSERT_EQ(output->at(0, 0, 2), 857);

  ASSERT_EQ(output->at(0, 1, 0), 99);
  ASSERT_EQ(output->at(0, 1, 1), 81);
  ASSERT_EQ(output->at(0, 1, 2), 857);
}