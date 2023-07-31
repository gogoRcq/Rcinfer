//
// Created by fss on 23-1-15.
//

#include "data/Tensor.h"
#include "operator/ExpressionOperator.h"
#include "runtime/PraseExpression.h"
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "layer/ExpressionLayer.h"
#include "LayerRegister.h"




typedef rq::Tensor<float> ftensor;

TEST(test_expression, add) {
    using namespace rq;
    const std::string &expr = "add(@0,@1)";
    std::shared_ptr<ExpressionOperator> expression_op = std::make_shared<ExpressionOperator>(expr);
    std::shared_ptr<Layer<float>> expr_layer = LayerRegister<float>::creatorLayer(expression_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    std::vector<std::shared_ptr<ftensor >> outputs;

    int batch_size = 4;
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(224, 224, 3);
        input->fill(1.f);
        inputs.push_back(input);
    }

    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(224, 224, 3);
        input->fill(2.f);
        inputs.push_back(input);
    }

    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> output = std::make_shared<ftensor>(224, 224, 3);
        outputs.push_back(output);
    }
    expr_layer->forwards(inputs, outputs);
    for (int i = 0; i < batch_size; ++i) {
        const auto &result = outputs.at(i);
        for (int j = 0; j < result->size(); ++j) {
            ASSERT_EQ(result->index(j), 3.f);
        }
    }
}

TEST(test_expression, complex) {
    using namespace rq;
    const std::string &expr = "add(mul(@0,@1),@2)";
    std::shared_ptr<ExpressionOperator> expression_op = std::make_shared<ExpressionOperator>(expr);
    std::shared_ptr<Layer<float>> expr_layer = LayerRegister<float>::creatorLayer(expression_op);
    std::vector<std::shared_ptr<ftensor >> inputs;
    std::vector<std::shared_ptr<ftensor >> outputs;

    int batch_size = 4;
    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(1.f);
        inputs.push_back(input);
    }

    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(2.f);
        inputs.push_back(input);
    }

    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
        input->fill(3.f);
        inputs.push_back(input);
    }

    for (int i = 0; i < batch_size; ++i) {
        std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
        outputs.push_back(output);
    }
    expr_layer->forwards(inputs, outputs);
    for (int i = 0; i < batch_size; ++i) {
        const auto &result = outputs.at(i);
        for (int j = 0; j < result->size(); ++j) {
            ASSERT_EQ(result->index(j), 5.f);
        }
    }
}

