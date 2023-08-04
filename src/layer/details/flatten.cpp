#include "details/flatten.h"
#include "common.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "layer/abstract/rcLayerRegister.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <memory>
#include <type_traits>


namespace rq {

template<class T>
InferStatus rcFlattenLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                        std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input of conv is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    int start_dim = this->startDim;
    int end_dim = this->endDim;
    int total_dims =4; // NCHW

    if (start_dim < 0) {
        start_dim = total_dims + start_dim;
    }
    if (end_dim < 0) {
        end_dim = total_dims + end_dim;
    }

    end_dim -= 1;
    start_dim -= 1;
    CHECK(end_dim > start_dim) << "End dim must greater than start dim";
    CHECK(end_dim <= 2 && start_dim >= 0) << "end dim must less than two and start dim must greater than zero";

    const uint32_t batch_size = inputs.size();
    #pragma omp parallel for num_threads(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<T>> &input = inputs.at(i);
        if (input == nullptr || input->empty()) {
            LOG(ERROR) << "The input feature map of flatten layer is empty";
            return InferStatus::rInferFailedInputEmpty;
        }
        std::vector<uint32_t> shapes = {input->channels(), input->rows(), input->cols()};
        
        uint32_t elements_size = 1;
        for (int s = start_dim; s <= end_dim; ++s) {
            elements_size *= shapes.at(s);
        }

        std::shared_ptr<Tensor<float>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = input->clone();
            outputs.at(i) = output;
        } else {
            memcpy(output->data().memptr(), input->data().memptr(), sizeof(float) * input->size());
        }
        if (start_dim == 0 && end_dim == 2) {
            output->ReRawView({elements_size}); 
        } else if (start_dim == 1 && end_dim == 2) {
            uint32_t channels = input->channels();
            output->ReRawView({channels, elements_size});
        } else if (start_dim == 0 && end_dim == 1) {
            uint32_t cols = input->cols();
            output->ReRawView({elements_size, cols});
        } else {
            LOG(FATAL) << "Wrong flatten dim: "
                       << "start dim: " << start_dim << " end dim: " << end_dim;
        }
    }
    return InferStatus::rInferSuccess;
}


template<class T>
ParseParamAttrStatus rcFlattenLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                        std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr);
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = op->params;

    if (params.find("start_dim") == params.end()) {
        LOG(ERROR) << "Can not find start_dim parameter";
        return ParseParamAttrStatus::rParameterMissingScale;
    }
    const std::shared_ptr<RuntimeParamInt> start_dim = std::dynamic_pointer_cast<RuntimeParamInt>(params["start_dim"]);
    if (!start_dim) {
        LOG(ERROR) << "Can not find start_dim parameter";
        return ParseParamAttrStatus::rParameterMissingScale;
    }

    if (params.find("end_dim") == params.end()) {
        LOG(ERROR) << "Can not find end_dim parameter";
        return ParseParamAttrStatus::rParameterMissingScale;
    }
    const std::shared_ptr<RuntimeParamInt> end_dim = std::dynamic_pointer_cast<RuntimeParamInt>(params["end_dim"]);
    if (!end_dim) {
        LOG(ERROR) << "Can not find end_dim parameter";
        return ParseParamAttrStatus::rParameterMissingScale;
    }

    layer = std::make_shared<rcFlattenLayer<T>>(start_dim->value, end_dim->value);
    return ParseParamAttrStatus::rParameterAttrParseSuccess;
}

INSTALLCLASS(rcFlattenLayer);
RCREGISTER_CREATOR(flatten, "torch.flatten", rcFlattenLayer);

}