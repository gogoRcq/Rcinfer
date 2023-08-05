// #include <gtest/gtest.h>
// #include <glog/logging.h>
// #include <memory>
// #include <sys/_types/_int32_t.h>
// #include <vector>
// #include "details/convolution.h"
// #include "runtime/RuntimeAttr.h"
// #include "runtime/RuntimeDataType.h"
// #include "runtime/RuntimeParam.h"
// #include "layer/abstract/rcLayerRegister.h"


// typedef rq::Tensor<float> ftensor;

// // 单卷积单通道
// TEST(test_layer, conv1) {
//     using namespace rq;
//     LOG(INFO) << "My convolution test!";
//     std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
//     runtime_op->name = "conv";
//     runtime_op->type = "nn.Conv2d";
//     std::unordered_map<std::string, std::shared_ptr<RuntimeParam>> params;
//     auto dilation = std::make_shared<RuntimeParamIntArray>();
//     dilation->value = {1, 1};
//     params.insert({"dilation", dilation});

//     auto in_channels = std::make_shared<RuntimeParamInt>();
//     in_channels->value = 1;
//     params.insert({"in_channels", in_channels});

//     auto out_channels = std::make_shared<RuntimeParamInt>();
//     out_channels->value = 1;
//     params.insert({"out_channels", out_channels});

//     auto groups = std::make_shared<RuntimeParamInt>();
//     groups->value = 1;
//     params.insert({"groups", groups});

//     auto bias = std::make_shared<RuntimeParamBool>();
//     bias->value = false;
//     params.insert({"bias", bias});

//     auto padding_mode = std::make_shared<RuntimeParamString>();
//     padding_mode->value = "zeros";
//     params.insert({"padding_mode", padding_mode});

//     auto padding = std::make_shared<RuntimeParamIntArray>();
//     padding->value = {0, 0};
//     params.insert({"padding", padding});

//     auto stride = std::make_shared<RuntimeParamIntArray>();
//     stride->value = {1, 1};
//     params.insert({"stride", stride});

//     auto kernel_size = std::make_shared<RuntimeParamIntArray>();
//     kernel_size->value = {3, 3};
//     params.insert({"kernel_size", kernel_size});
//     runtime_op->params = params;

//     std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<float>>>& attributes = runtime_op->attributes;
//     std::vector<float> values;
//     for (int i = 0; i < 3; ++i) {
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//     }

//     auto weightsAtrr = std::make_shared<RuntimeAttr<float>>();
//     std::vector<char>& weightData = weightsAtrr->weightData;
//     char *start = (char*)values.data();
//     for (int i = 0; i < values.size() * sizeof(float); ++i) {
//         weightData.emplace_back(*(start + i));
//     }
//     weightsAtrr->shape = {(int32_t)(values.size() * sizeof(float))};
//     weightsAtrr->dataType = RuntimeDataType::rTypeFloat32;
//     attributes.insert({"weight", weightsAtrr});

//     std::shared_ptr<rcLayer<float>> conv_layer = rcLayerRegister<float>::CreateLayer(runtime_op);

//     std::vector<std::shared_ptr<ftensor >> inputs;
//     arma::fmat input_data = "1,2,3,4;"
//                             "5,6,7,8;"
//                             "7,8,9,10;"
//                             "11,12,13,14";
//     std::shared_ptr<ftensor> input = std::make_shared<ftensor>(4, 4, 1);
//     input->at(0) = input_data;
//     LOG(INFO) << "input:";
//     input->show();
//     // 权重数据和输入数据准备完毕
//     inputs.push_back(input);
//     std::vector<std::shared_ptr<ftensor >> outputs(1);

//     conv_layer->forwards(inputs, outputs);
//     LOG(INFO) << "result: ";
//     for (int i = 0; i < outputs.size(); ++i) {
//         outputs.at(i)->show();
//     }
// }

// // 多卷积多通道
// TEST(test_layer, conv2) {
//     using namespace rq;
//     LOG(INFO) << "My convolution test!";
//     std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
//     runtime_op->name = "conv";
//     runtime_op->type = "nn.Conv2d";
//     std::unordered_map<std::string, std::shared_ptr<RuntimeParam>> params;
//     auto dilation = std::make_shared<RuntimeParamIntArray>();
//     dilation->value = {1, 1};
//     params.insert({"dilation", dilation});

//     auto in_channels = std::make_shared<RuntimeParamInt>();
//     in_channels->value = 3;
//     params.insert({"in_channels", in_channels});

//     auto out_channels = std::make_shared<RuntimeParamInt>();
//     out_channels->value = 3;
//     params.insert({"out_channels", out_channels});

//     auto groups = std::make_shared<RuntimeParamInt>();
//     groups->value = 1;
//     params.insert({"groups", groups});

//     auto bias = std::make_shared<RuntimeParamBool>();
//     bias->value = false;
//     params.insert({"bias", bias});

//     auto padding_mode = std::make_shared<RuntimeParamString>();
//     padding_mode->value = "zeros";
//     params.insert({"padding_mode", padding_mode});

//     auto padding = std::make_shared<RuntimeParamIntArray>();
//     padding->value = {0, 0};
//     params.insert({"padding", padding});

//     auto stride = std::make_shared<RuntimeParamIntArray>();
//     stride->value = {1, 1};
//     params.insert({"stride", stride});

//     auto kernel_size = std::make_shared<RuntimeParamIntArray>();
//     kernel_size->value = {3, 3};
//     params.insert({"kernel_size", kernel_size});
//     runtime_op->params = params;

//     std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<float>>>& attributes = runtime_op->attributes;
//     std::vector<float> values;
//     for (int i = 0; i < 3; ++i) {
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//     }
//     for (int i = 0; i < 3; ++i) {
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//     }
//     for (int i = 0; i < 3; ++i) {
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//     }

//     auto weightsAtrr = std::make_shared<RuntimeAttr<float>>();
//     std::vector<char>& weightData = weightsAtrr->weightData;
//     char *start = (char*)values.data();
//     for (int i = 0; i < values.size() * sizeof(float) * 3; ++i) {
//         weightData.emplace_back(*(start + i % (values.size() * sizeof(float))));
//     }
//     weightsAtrr->shape = {(int32_t)(values.size() * sizeof(float) * 3)};
//     weightsAtrr->dataType = RuntimeDataType::rTypeFloat32;
//     attributes.insert({"weight", weightsAtrr});

//     std::shared_ptr<rcLayer<float>> conv_layer = rcLayerRegister<float>::CreateLayer(runtime_op);

//     std::vector<std::shared_ptr<ftensor >> inputs;
//     arma::fmat input_data = "1,2,3,4;"
//                             "5,6,7,8;"
//                             "7,8,9,10;"
//                             "11,12,13,14";
//     std::shared_ptr<ftensor> input = std::make_shared<ftensor>(4, 4, 3);
//     input->at(0) = input_data;
//     input->at(1) = input_data;
//     input->at(2) = input_data;

//     LOG(INFO) << "input:";
//     input->show();
//     // 权重数据和输入数据准备完毕
//     inputs.push_back(input);
//     std::vector<std::shared_ptr<ftensor >> outputs(1);

//     conv_layer->forwards(inputs, outputs);
//     LOG(INFO) << "result: ";
//     for (int i = 0; i < outputs.size(); ++i) {
//         outputs.at(i)->show();
//     }
// }

// // 单卷积单通道
// TEST(test_layer, conv3) {
//     using namespace rq;
//     LOG(INFO) << "My convolution test!";
//     std::shared_ptr<RuntimeOperator<float>> runtime_op = std::make_shared<RuntimeOperator<float>>();
//     runtime_op->name = "conv";
//     runtime_op->type = "nn.Conv2d";
//     std::unordered_map<std::string, std::shared_ptr<RuntimeParam>> params;
//     auto dilation = std::make_shared<RuntimeParamIntArray>();
//     dilation->value = {1, 1};
//     params.insert({"dilation", dilation});

//     auto in_channels = std::make_shared<RuntimeParamInt>();
//     in_channels->value = 1;
//     params.insert({"in_channels", in_channels});

//     auto out_channels = std::make_shared<RuntimeParamInt>();
//     out_channels->value = 1;
//     params.insert({"out_channels", out_channels});

//     auto groups = std::make_shared<RuntimeParamInt>();
//     groups->value = 1;
//     params.insert({"groups", groups});

//     auto bias = std::make_shared<RuntimeParamBool>();
//     bias->value = true;
//     params.insert({"bias", bias});

//     auto padding_mode = std::make_shared<RuntimeParamString>();
//     padding_mode->value = "zeros";
//     params.insert({"padding_mode", padding_mode});

//     auto padding = std::make_shared<RuntimeParamIntArray>();
//     padding->value = {1, 1};
//     params.insert({"padding", padding});

//     auto stride = std::make_shared<RuntimeParamIntArray>();
//     stride->value = {1, 1};
//     params.insert({"stride", stride});

//     auto kernel_size = std::make_shared<RuntimeParamIntArray>();
//     kernel_size->value = {3, 3};
//     params.insert({"kernel_size", kernel_size});
//     runtime_op->params = params;

//     std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<float>>>& attributes = runtime_op->attributes;
//     std::vector<float> values;
//     for (int i = 0; i < 3; ++i) {
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//         values.push_back(float(i + 1));
//     }
//     auto weightsAtrr = std::make_shared<RuntimeAttr<float>>();
//     std::vector<char>& weightData = weightsAtrr->weightData;
//     char *start = (char*)values.data();
//     for (int i = 0; i < values.size() * sizeof(float); ++i) {
//         weightData.emplace_back(*(start + i));
//     }
//     weightsAtrr->shape = {(int32_t)(values.size() * sizeof(float))};
//     weightsAtrr->dataType = RuntimeDataType::rTypeFloat32;
//     attributes.insert({"weight", weightsAtrr});

//     auto biasAtrr = std::make_shared<RuntimeAttr<float>>();
//     auto& biasData = biasAtrr->weightData;
//     std::vector<float> biasd(1, 1.0f);
//     char *bstart = (char*)biasd.data();
//     for (int i = 0; i < biasd.size() * sizeof(float); ++i) {
//         biasData.emplace_back(*(bstart + i));
//     }
//     biasAtrr->dataType = RuntimeDataType::rTypeFloat32;
//     biasAtrr->shape = {1};
//     attributes.insert({"bias", biasAtrr});

//     std::shared_ptr<rcLayer<float>> conv_layer = rcLayerRegister<float>::CreateLayer(runtime_op);
//     // 单个卷积核的情况;
    
//     std::vector<std::shared_ptr<ftensor >> inputs;
//     arma::fmat input_data = "6,7;"
//                             "7,8,";
//     std::shared_ptr<ftensor> input = std::make_shared<ftensor>(2, 2, 1);
//     input->at(0) = input_data;
//     LOG(INFO) << "input:";
//     input->show();
//     // 权重数据和输入数据准备完毕
//     inputs.push_back(input);
//     std::vector<std::shared_ptr<ftensor >> outputs(1);

//     conv_layer->forwards(inputs, outputs);
//     LOG(INFO) << "result: ";
//     for (int i = 0; i < outputs.size(); ++i) {
//         outputs.at(i)->show();
//     }
// }