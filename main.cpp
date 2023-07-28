#include "glog/logging.h"
#include <armadillo>
#include "data/Tensor.h"
#include "operator/ReluOperator.h"
#include "layer/ReluLayer.h"
#include "LayerRegister.h"

int main(int argc, char const *argv[]) {
    using namespace rq;
    float thresh = 0.f;
    std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator<float>>(thresh);
    std::shared_ptr<Layer<float>> relu_layer = LayerRegister<float>::creatorLayer(relu_op);
}
