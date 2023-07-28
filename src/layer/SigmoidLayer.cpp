#include "layer/SigmoidLayer.h"
#include "common.h"

namespace rq {

template<class T>
std::shared_ptr<Layer<T>> SigmoidLayer<T>::creatorInstance(const std::shared_ptr<Operator>& op) {
    return std::make_shared<SigmoidLayer<T>>(op);
}

template<class T>
void SigmoidLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs, std::vector<std::shared_ptr<Tensor<T>>> &outputs){
    CHECK(this->op != NULL);
    CHECK(this->op->opType == OperatorType::rOperatorSigmoid);
    CHECK(!inputs.empty());

    const uint32_t batch_size = inputs.size();
    for (size_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<T>> &input_data = inputs[i];
        std::shared_ptr<Tensor<T>> output_data = input_data->clone();
        output_data->data().transform([&](T val){
            return (T)(1.0 / (1.0 + (T)std::exp(-val)));
        });
        outputs.push_back(output_data);
    }
}

INSTALLCLASS(SigmoidLayer);
REGISTER_CREATOR(sigmoid, OperatorType::rOperatorSigmoid, SigmoidLayer);

}