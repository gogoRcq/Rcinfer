#include <gtest/gtest.h>
#include <glog/logging.h>
#include "layer/ConvLayer.h"
#include "operator/ConvOperator.h"

typedef rq::Tensor<float> ftensor;

// 单卷积单通道
TEST(test_layer, conv1) {
    using namespace rq;
    LOG(INFO) << "My convolution test!";
    std::shared_ptr<ConvOperator<float>> conv_op = std::make_shared<ConvOperator<float>>(false, 1, 1, 1, 0, 0);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 3; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 1);
    weight1->fill(values);
    LOG(INFO) << "weight:";
    weight1->show();
    // 设置权重
    std::vector<std::shared_ptr<ftensor>> weights;
    weights.push_back(weight1);
    conv_op->setWeights(weights);
    std::shared_ptr<Layer<float>> conv_layer = LayerRegister<float>::creatorLayer(conv_op);

    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4;"
                            "5,6,7,8;"
                            "7,8,9,10;"
                            "11,12,13,14";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(4, 4, 1);
    input->at(0) = input_data;
    LOG(INFO) << "input:";
    input->show();
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    std::vector<std::shared_ptr<ftensor >> outputs(1);

    conv_layer->forwards(inputs, outputs);
    LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i) {
        outputs.at(i)->show();
    }
}

// 多卷积多通道
TEST(test_layer, conv2) {
    using namespace rq;
    LOG(INFO) << "My convolution test!";
    std::shared_ptr<ConvOperator<float>> conv_op = std::make_shared<ConvOperator<float>>(false, 1, 1, 1, 0, 0);
    // 单个卷积核的情况
    std::vector<float> values;

    arma::fmat weight_data = "1 ,1, 1 ;"
                            "2 ,2, 2;"
                            "3 ,3, 3;";
    // 初始化三个卷积核
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 3);
    weight1->at(0) = weight_data;
    weight1->at(1) = weight_data;
    weight1->at(2) = weight_data;

    std::shared_ptr<ftensor> weight2 = weight1->clone();
    std::shared_ptr<ftensor> weight3 = weight1->clone();

    LOG(INFO) << "weight:";
    weight1->show();
    // 设置权重
    std::vector<std::shared_ptr<ftensor >> weights;
    weights.push_back(weight1);
    weights.push_back(weight2);
    weights.push_back(weight3);

    conv_op->setWeights(weights);
    std::shared_ptr<Operator> op = conv_op;

    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "1,2,3,4;"
                            "5,6,7,8;"
                            "7,8,9,10;"
                            "11,12,13,14";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(4, 4, 3);
    input->at(0) = input_data;
    input->at(1) = input_data;
    input->at(2) = input_data;

    LOG(INFO) << "input:";
    input->show();
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    std::shared_ptr<Layer<float>> layer = LayerRegister<float>::creatorLayer(conv_op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);

    layer->forwards(inputs, outputs);
    LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i) {
        outputs.at(i)->show();
    }
}

// 单卷积单通道
TEST(test_layer, conv3) {
    using namespace rq;
    LOG(INFO) << "My convolution test!";
    std::shared_ptr<ConvOperator<float>> conv_op = std::make_shared<ConvOperator<float>>(true, 1, 1, 1, 1, 1);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 3; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 1);
    weight1->fill(values);
    LOG(INFO) << "weight:";
    weight1->show();
    // 设置权重
    std::vector<std::shared_ptr<ftensor>> weights;
    std::vector<std::shared_ptr<ftensor>> bias;
    std::shared_ptr<ftensor> bs = std::make_shared<ftensor>(1, 1, 1);
    bs->fill(1.0f);
    bias.push_back(bs);
    conv_op->setBias(bias);
    weights.push_back(weight1);
    conv_op->setWeights(weights);
    std::shared_ptr<Layer<float>> conv_layer = LayerRegister<float>::creatorLayer(conv_op);

    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = "6,7;"
                            "7,8,";
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(2, 2, 1);
    input->at(0) = input_data;
    LOG(INFO) << "input:";
    input->show();
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    std::vector<std::shared_ptr<ftensor >> outputs(1);

    conv_layer->forwards(inputs, outputs);
    LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i) {
        outputs.at(i)->show();
    }
}