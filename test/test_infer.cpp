#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "details/softmax.h"
#include "glog/logging.h"
#include "runtime/RuntimeGraph.h"
#include "data/LoadData.h"
#include "opencv4/opencv2/opencv.hpp"
#include "data/Tensor.h"

std::shared_ptr<rq::Tensor<float>> PreProcessImage(const cv::Mat& image) {
    using namespace rq;
    assert(!image.empty());
    // 调整输入大小
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(224, 224));

    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);

    rgb_image.convertTo(rgb_image, CV_32FC3);
    std::vector<cv::Mat> split_images;
    cv::split(rgb_image, split_images);
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t input_c = 3;
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_h, input_w, input_c);

    uint32_t index = 0;
    for (const auto& split_image : split_images) {
        assert(split_image.total() == input_w * input_h);
        const cv::Mat& split_image_t = split_image.t();
        memcpy(input->at(index).memptr(), split_image_t.data,
            sizeof(float) * split_image.total());
        index += 1;
    }

    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;

    float var_r = 0.229f;
    float var_g = 0.224f;
    float var_b = 0.225f;
    assert(input->channels() == 3);
    input->data() = input->data() / 255.f;
    input->at(0) = (input->at(0) - mean_r) / var_r;
    input->at(1) = (input->at(1) - mean_g) / var_g;
    input->at(2) = (input->at(2) - mean_b) / var_b;
    return input;
}

TEST(test_infer, easy) {
    using namespace rq;
    const std::string& param_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/resnet18_batch1.pnnx.param";
    const std::string& weight_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/resnet18_batch1.pnnx.bin";
    RuntimeGraph<float> graph(param_path, weight_path);
    graph.Build("pnnx_input_0", "pnnx_output_0");
    LOG(INFO) << "-----------------start rqInfer-----------------";

    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(224, 224, 3);
    input1->fill(1.);
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.forward(inputs, false);
    ASSERT_EQ(outputs.size(), 1);

    const auto &output2 = CSVDataLoader<float>::loadData("/Users/rcq/home/cppprojs/Rcinfer/tmp/out.csv");
    const auto &output1 = outputs.front()->data().slice(0);
    ASSERT_EQ(output1.size(), output2->size());
    for (uint32_t s = 0; s < output1.size(); ++s) {
        ASSERT_LE(std::abs(output1.at(s) - output2->data().slice(0).at(s)), 1e-5);
    }
    LOG(INFO) << "data is same !!!!";
}

void getLabels(std::vector<std::string>& labels, const std::string& path) {
    std::ifstream inFile;
    inFile.open(path);
    CHECK(inFile) << "open error";
    std::string str;
    while (std::getline(inFile, str)) {
        labels.emplace_back(str);
    }
}

TEST(test_infer, resbet18) {
    using namespace rq;

    std::vector<std::string> labels;
    labels.reserve(1000);
    getLabels(labels, "../tmp/imagenet-classes.txt");

    std::string path = "../tmp/dog.jpg";
    cv::Mat image = cv::imread(path);
    // 图像预处理
    std::shared_ptr<Tensor<float>> input = PreProcessImage(image);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    const std::string& param_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/resnet18_batch1.pnnx.param";
    const std::string& weight_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/resnet18_batch1.pnnx.bin";
    RuntimeGraph<float> graph(param_path, weight_path);
    graph.Build("pnnx_input_0", "pnnx_output_0");
    LOG(INFO) << "-----------------start rqInfer-----------------";

    // 推理
    const std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.forward(inputs, false);
    const uint32_t batch_size = 1;
    // softmax
    rcSoftmaxLayer<float> softmax;

    std::vector<std::shared_ptr<Tensor<float>>> outputs_softmax(batch_size);
    softmax.forwards(outputs, outputs_softmax);
    for (int i = 0; i < outputs_softmax.size(); ++i) {
        const std::shared_ptr<Tensor<float>>& output_tensor = outputs_softmax.at(i);
        assert(output_tensor->size() == 1 * 1000);
        // 找到类别概率最大的种类
        float max_prob = -1;
        int max_index = -1;
        for (int j = 0; j < output_tensor->size(); ++j) {
            float prob = output_tensor->index(j);
            if (max_prob <= prob) {
                max_prob = prob;
                max_index = j;
            }
        }
        LOG(INFO) << "***************************";
        LOG(INFO) << "class with max prob is : " << max_prob << " index: " << max_index << " label: " << labels[max_index];
        LOG(INFO) << "***************************";;
    }
}