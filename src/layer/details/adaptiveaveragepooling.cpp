#include "details/adaptiveaveragepooling.h"
#include "glog/logging.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <cmath>
#include <memory>


namespace rq {

template<class T>
InferStatus AdaptiveAveragePoolingLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                                     std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input of conv is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    if (outputW <= 0 || outputH <= 0) {
        LOG(ERROR) << "The output size of adaptive pooling is less than zero";
        return InferStatus::rInferFailedOutputSizeError;
    }

    const uint32_t batchSize = inputs.size();
    for (int i = 0; i < batchSize; ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        const std::shared_ptr<Tensor<T>>& output = outputs[i];
        if (input == nullptr || input->empty()) {
            LOG(ERROR) << "The input feature map of adaptive pooling layer is empty";
            return InferStatus::rInferFailedInputEmpty;
        }
        if (output != nullptr && !output->empty()) {
            if (output->rows() != outputH || output->cols() != outputW) {
                LOG(ERROR) << "The output size of adaptive pooling is not adapting";
                return InferStatus::rInferFailedOutputSizeError;
            }
        }
    }
    #pragma omp parallel for num_threads(batchSize)
    for (int i = 0; i < batchSize; ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        const uint32_t inputH = input->rows();
        const uint32_t inputW = input->cols();
        const uint32_t inputC = input->channels();
        const uint32_t strideH = uint32_t(std::floor(inputH / outputH));
        const uint32_t strideW = uint32_t(std::floor(inputW/ outputW));
        CHECK(strideH > 0 && strideW > 0) << "error stride num";
        const uint32_t poolingH = inputH - (outputH - 1) * strideH;
        const uint32_t poolingW = inputW - (outputW - 1) * strideW;
        CHECK(poolingW > 0 && poolingH > 0) << "The pooling parameter is set incorrectly";

        std::shared_ptr<Tensor<T>> output = outputs[i];
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<T>>(outputH, outputW, inputC);
            outputs.at(i) = output;
        }

        CHECK(output->rows() == outputH && output->cols() == outputW && output->channels() == inputC) << "error output shape";

        for (int ic = 0; ic < inputC; ++ic) {
            const arma::Mat<T> inputchannel = input->at(ic);
            arma::Mat<T>& out_channel = output->at(ic);
            uint32_t up_row = 0, left_col = 0, down_row = poolingH - 1, right_col = poolingW - 1;
            for (uint32_t row = 0; row < outputH; ++row) {
                for (uint32_t col = 0; col < outputW; ++col) {
                    const arma::Mat<T>& subM = inputchannel.submat(up_row + row * strideH, left_col + col * strideW, 
                                                                   down_row + row * strideH, right_col + col * strideW);
                    out_channel.at(row, col) = arma::mean(arma::mean(subM));
                }
            }
        }
    }
    return InferStatus::rInferSuccess;                     
}

template<class T>
ParseParamAttrStatus AdaptiveAveragePoolingLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                                     std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr);
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = op->params;
    CHECK(!params.empty()) << "Operator parameter is empty";

    if (params.find("output_size") == params.end()) {
        LOG(ERROR) << "Can not find the output_size parameter";
        return ParseParamAttrStatus::rParameterMissingOutHW;
    }
    const std::shared_ptr<RuntimeParamIntArray> output_size = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["output_size"]);
    if (output_size == nullptr || output_size->value.size() != 2) {
        LOG(ERROR) << "Can not find the output_size parameter";
        return ParseParamAttrStatus::rParameterMissingOutHW;
    }

    layer = std::make_shared<AdaptiveAveragePoolingLayer<T>>(output_size->value[0], output_size->value[1]);
    return ParseParamAttrStatus::rParameterAttrParseSuccess;
}

INSTALLCLASS(AdaptiveAveragePoolingLayer);
RCREGISTER_CREATOR(adaptivepooling, "nn.AdaptiveAvgPool2d", AdaptiveAveragePoolingLayer);

}