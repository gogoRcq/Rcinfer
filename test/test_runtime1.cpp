//
// Created by fss on 23-1-7.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/RuntimeGraph.h"

TEST(test_runtime, runtime1) {
    using namespace rq;
    const std::string &param_path = "../tmp/test.pnnx.param";
    const std::string &bin_path = "../tmp/test.pnnx.bin";
    RuntimeGraph<float> graph(param_path, bin_path);
    graph.init();
    graph.Build("pnnx_input_0", "pnnx_output_0");
    const auto operators = graph.operators();
    for (const auto &operator_ : operators) {
        LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
    }
}
