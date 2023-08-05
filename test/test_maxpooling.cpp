//
// Created by fss on 23-1-1.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include "details/maxpooling.h"
#include "runtime/RuntimeOperator.h"
#include "layer/abstract/rcLayerRegister.h"
#include "runtime/RuntimeParam.h"


TEST(test_layer, forward_maxpooling1) {
    using namespace rq;

    std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
    runtime_op->name = "maxpooling";
    runtime_op->type = "nn.MaxPool2d";
    auto stride = std::make_shared<RuntimeParamIntArray>();
    stride->value = {1, 3};
    auto padding = std::make_shared<RuntimeParamIntArray>();
    padding->value = {0, 0};
    auto pooling = std::make_shared<RuntimeParamIntArray>();
    pooling->value = {3, 3};
    runtime_op->params.insert({"stride", stride});
    runtime_op->params.insert({"padding", padding});
    runtime_op->params.insert({"kernel_size", pooling});

    std::shared_ptr<rcLayer<float>> max_layer = rcLayerRegister<float>::CreateLayer(runtime_op);
    CHECK(max_layer != nullptr);

    arma::fmat input_data = "71 22 63 94  65 16 75 58  9  11;"
                            "12 13 99 31 -31 55 99 857 12 511;"
                            "52 15 19 81 -61 15 49 67  12 41;"
                            "41 41 61 21 -15 15 10 13  51 55;";
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_data.n_rows, input_data.n_cols, 1);
    input->at(0) = input_data;
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
    inputs.push_back(input);
    // LOG(INFO) << input->data();
    max_layer->forwards(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
    const auto &output = outputs.at(0);
    // LOG(INFO) << output->data();
    // LOG(INFO) << input->data();

    ASSERT_EQ(output->rows(), 2);
    ASSERT_EQ(output->cols(), 3);

    ASSERT_EQ(output->at(0, 0, 0), 99);
    ASSERT_EQ(output->at(0, 0, 1), 94);
    ASSERT_EQ(output->at(0, 0, 2), 857);

    ASSERT_EQ(output->at(0, 1, 0), 99);
    ASSERT_EQ(output->at(0, 1, 1), 81);
    ASSERT_EQ(output->at(0, 1, 2), 857);
}