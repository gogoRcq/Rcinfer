#include "relu.h"
#include "common.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "layer/abstract/rcLayerRegister.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <memory>

namespace rq {
template<class T>
InferStatus rcReluLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input of relu is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    const uint32_t batch_size = inputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        const std::shared_ptr<Tensor<T>>& output = outputs[i];
        if (input == nullptr || input->empty()) {
            LOG(ERROR) << "Input of relue is empty";
            return InferStatus::rInferFailedInputEmpty;
        }
        if (output != nullptr && !output->empty()) {
            if (output->shapes() != input->shapes()) {
                LOG(ERROR) << "The input and output size is not adapting";
                return InferStatus::rInferFailedInputOutSizeAdaptingError;
            }
        }
    }

    #pragma omp parallel for num_threads(batch_size)
    for (size_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<T>> &input = inputs[i];
        std::shared_ptr<Tensor<T>> output = outputs[i];
        if (output == nullptr || output->empty()) {
            CHECK(input->shapes().size() == 3);
            output = std::make_shared<Tensor<T>>(input->shapes()[0], input->shapes()[1], input->shapes()[2]);
            outputs[i] = output;
        }
        CHECK(output->shapes() == input->shapes());
        output->setdata(input->data());
        output->data().transform([&](T val){
            if (val < (T)0) return (T)0;
            else return val;
        });
    }   
    return InferStatus::rInferSuccess;                  
}

template<class T>
ParseParamAttrStatus rcReluLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                     std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr);
    layer = std::make_shared<rcReluLayer<T>>();
    return ParseParamAttrStatus::rParameterAttrParseSuccess;
}

INSTALLCLASS(rcReluLayer);
RCREGISTER_CREATOR(relu, "nn.ReLU", rcReluLayer);

}