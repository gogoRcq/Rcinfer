#include <glog/logging.h>
#include <gtest/gtest.h>
#include "layer/abstract/rcLayerRegister.h"
#include "details/flatten.h"
#include "runtime/RuntimeOperator.h"

TEST(test_layer, flatten) {
    using namespace rq;

    std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
    runtime_op->name = "flatten";
    runtime_op->type = "torch.flatten";
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = runtime_op->params;

    auto start_dim = std::make_shared<RuntimeParamInt>();
    start_dim->value = 1;
    params.insert({"start_dim", start_dim});

    auto end_dim = std::make_shared<RuntimeParamInt>();
    end_dim->value = -1;
    params.insert({"end_dim", end_dim});

    std::shared_ptr<rcLayer<float>> flatten_layer = rcLayerRegister<float>::CreateLayer(runtime_op);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs(2);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, 2, 1);
    input->fill(1.0f);
    inputs.emplace_back(input);
    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(2, 2, 1);
    input2->fill(2.0f);
    inputs.emplace_back(input2);
    auto op1 = std::make_shared<Tensor<float>>(4, 1, 1);
    auto op2  = std::make_shared<Tensor<float>>(4, 1, 1);
    outputs[0] = op1;
    outputs[1] = op2;
    flatten_layer->forwards(inputs, outputs);
    LOG(INFO) << outputs[0]->data();
    LOG(INFO) << outputs[1]->data();
}