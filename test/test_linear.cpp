#include <glog/logging.h>
#include <gtest/gtest.h>
#include <vector>
#include "data/Tensor.h"
#include "layer/abstract/rcLayerRegister.h"
#include "details/linear.h"
#include "runtime/RuntimeDataType.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"

TEST(test_layer, forward_linear) {
    using namespace rq;
    std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
    runtime_op->name = "linear";
    runtime_op->type = "nn.Linear";
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = runtime_op->params;
    auto bias = std::make_shared<RuntimeParamBool>();
    bias->value = true;
    params.insert({"bias", bias});

    std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<float>>>& attributes = runtime_op->attributes;
    std::vector<float> wvalues(50, 1);
    auto weightsAtrr = std::make_shared<RuntimeAttr<float>>();
    weightsAtrr->dataType = RuntimeDataType::rTypeFloat32;
    weightsAtrr->shape = {10, 5};
    auto& weightdata = weightsAtrr->weightData;
    char *wstart = (char *)wvalues.data();
    for (int i = 0; i < wvalues.size() * sizeof(float); ++i) {
        weightdata.emplace_back(*(wstart + i));
    }
    attributes.insert({"weight", weightsAtrr});

    std::vector<float> bvalues(10, 1);
    auto biasAtrr = std::make_shared<RuntimeAttr<float>>();
    biasAtrr->dataType = RuntimeDataType::rTypeFloat32;
    biasAtrr->shape = {10};
    auto& biasdata = biasAtrr->weightData;
    char *bstart = (char *)bvalues.data();
    for (int i = 0; i < bvalues.size() * sizeof(float); ++i) {
        biasdata.emplace_back(*(bstart + i));
    }
    attributes.insert({"bias", biasAtrr});

    std::shared_ptr<rcLayer<float>> linear_layer = rcLayerRegister<float>::CreateLayer(runtime_op);
    
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs(2);
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(5, 1, 1);
    input->fill(1.0f);
    inputs.emplace_back(input);
    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(5, 1, 1);
    input2->fill(2.0f);
    inputs.emplace_back(input2);
    linear_layer->forwards(inputs, outputs);
    LOG(INFO) << outputs[0]->data();
    LOG(INFO) << outputs[1]->data();
}