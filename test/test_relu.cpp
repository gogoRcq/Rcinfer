// #include <glog/logging.h>
// #include <gtest/gtest.h>
// #include "layer/abstract/rcLayerRegister.h"
// #include "details/relu.h"
// #include "runtime/RuntimeOperator.h"

// TEST(test_layer, forward_relu1) {
//     using namespace rq;
//     // 初始化一个relu operator 并设置属性
//     std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
//     runtime_op->name = "relu";
//     runtime_op->type = "nn.ReLU";

//     std::shared_ptr<rcLayer<float>> layer = rcLayerRegister<float>::CreateLayer(runtime_op);

//     // 有三个值的一个tensor<float>
//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 1, 1);
//     input->index(0) = -1.f; //output对应的应该是0
//     input->index(1) = -2.f; //output对应的应该是0
//     input->index(2) = 3.f; //output对应的应该是3
//     LOG(INFO) << input->data();
//     // 主要第一个算子，经典又简单，我们这里开始！

//     std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理
//     std::vector<std::shared_ptr<Tensor<float>>> outputs(1); //放结果
//     inputs.push_back(input);
//     layer->forwards(inputs, outputs);
//     ASSERT_EQ(outputs.size(), 1);

//     LOG(INFO) << outputs[0]->data();

//     for (int i = 0; i < outputs.size(); ++i) {
//         ASSERT_EQ(outputs.at(i)->index(0), 0.f);
//         ASSERT_EQ(outputs.at(i)->index(1), 0.f);
//         ASSERT_EQ(outputs.at(i)->index(2), 3.f);
//     }
// }

// TEST(test_layer, forward_relu2) {
//     using namespace rq;
//     std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
//     runtime_op->name = "relu";
//     runtime_op->type = "nn.ReLU";

//     std::shared_ptr<rcLayer<float>> relu_layer = rcLayerRegister<float>::CreateLayer(runtime_op);

//     std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
//     input->index(0) = -1.f;
//     input->index(1) = -2.f;
//     input->index(2) = 3.f;
//     std::vector<std::shared_ptr<Tensor<float>>> inputs;
//     std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
//     inputs.push_back(input);
//     relu_layer->forwards(inputs, outputs);
//     ASSERT_EQ(outputs.size(), 1);
//     for (int i = 0; i < outputs.size(); ++i) {
//         ASSERT_EQ(outputs.at(i)->index(0), 0.f);
//         ASSERT_EQ(outputs.at(i)->index(1), 0.f);
//         ASSERT_EQ(outputs.at(i)->index(2), 3.f);
//     }
// }