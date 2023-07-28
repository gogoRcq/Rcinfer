#include "layer/ReluLayer.h"
#include "operator/Operator.h"
#include "common.h"

namespace rq {

template<class T>
std::shared_ptr<Layer<T>> ReluLayer<T>::creatorInstance(const std::shared_ptr<Operator> &op) {
    return std::make_shared<ReluLayer<T>>(op);
}

template<class T>
void ReluLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                            std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    CHECK(this->op != NULL);
    CHECK(this->op->opType == OperatorType::rOperatorRelu);
    CHECK(!inputs.empty());

    const uint32_t batch_size = inputs.size();
    const T thresh = op->getThresh();
    for (size_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<T>> &input_data = inputs[i];
        std::shared_ptr<Tensor<T>> output_data = input_data->clone();
        output_data->data().transform([&](T val){
            if (val < thresh) return (T)0;
            else return val;
        });
        outputs.push_back(output_data);
    }
}

INSTALLCLASS(ReluLayer);
REGISTER_CREATOR(relu, OperatorType::rOperatorRelu, ReluLayer);

}
