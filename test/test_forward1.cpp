//
// Created by fss on 23-2-21.
//
#include "runtime/RuntimeGraph.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

typedef rq::Tensor<float> ftensor;

typedef std::shared_ptr<rq::Tensor<float>> sftensor;

TEST(test_forward, forward1) {
  using namespace rq;
  const std::string &param_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/resnet18_hub.pnnx.param";
  const std::string &weight_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/resnet18_hub.pnnx.bin";
  RuntimeGraph<float> graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &operators = graph.operators();
  LOG(INFO) << "operator size: " << operators.size();
  uint32_t batch_size = 2;
  std::vector<sftensor> inputs(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(256, 256, 3);
    inputs.at(i)->fill(1.f);
  }
  const std::vector<sftensor> &outputs = graph.forward(inputs, true);
}

TEST(test_forward, forward2) {
  using namespace rq;
  const std::string &param_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/test.pnnx.param";
  const std::string &weight_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/test.pnnx.bin";
  RuntimeGraph<float> graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &operators = graph.operators();
  LOG(INFO) << "operator size: " << operators.size();
  uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(16, 16, 1);
    inputs.at(i)->fill(1.f);
  }
  const std::vector<sftensor> &outputs = graph.forward(inputs, true);
}

TEST(test_forward, forward3) {
  using namespace rq;
  const std::string &param_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/ten.pnnx.param";
  const std::string &weight_path = "/Users/rcq/home/cppprojs/Rcinfer/tmp/ten.pnnx.bin";
  RuntimeGraph<float> graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &operators = graph.operators();
  LOG(INFO) << "operator size: " << operators.size();
  uint32_t batch_size = 2;
  std::vector<sftensor> inputs(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(128, 128, 3);
    inputs.at(i)->fill(1.f);
  }
  const std::vector<sftensor> &outputs = graph.forward(inputs, true);
}