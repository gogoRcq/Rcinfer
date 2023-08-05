// #include <glog/logging.h>
// #include <gtest/gtest.h>
// #include "layer/abstract/rcLayerRegister.h"
// #include "details/adaptiveaveragepooling.h"
// #include "runtime/RuntimeOperator.h"

// TEST(test_layer, test_adp) {
//     using namespace rq;
//     std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
//     runtime_op->name = "adp";
//     runtime_op->type = "nn.AdaptiveAvgPool2d";
//     std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = runtime_op->params;
//     auto output_size = std::make_shared<RuntimeParamIntArray>();
//     output_size->value = {2, 2};
//     params.insert({"output_size", output_size});

//     std::shared_ptr<rcLayer<float>> adp_layer = rcLayerRegister<float>::CreateLayer(runtime_op);

//     std::vector<std::shared_ptr<Tensor<float>>> inputs;
//     std::vector<std::shared_ptr<Tensor<float>>> outputs(2);

//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 3, 2);
//     input->fill({1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 3, 3, 2, 2, 2, 5, 5, 5});
//     inputs.emplace_back(input);
//     auto input2 = input->clone();
//     inputs.emplace_back(input2);
//     adp_layer->forwards(inputs, outputs);
//     LOG(INFO) << outputs[0]->data();
//     LOG(INFO) << outputs[1]->data();
// }