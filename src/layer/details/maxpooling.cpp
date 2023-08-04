#include "details/maxpooling.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "layer/abstract/rcLayer.h"
#include "runtime/RuntimeParam.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <memory>
#include "common.h"
#include "layer/abstract/rcLayerRegister.h"

namespace rq {

template<class T>
InferStatus rcMaxPoolingLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input of maxpooling is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    uint32_t poolingH = this->poolingH;
    uint32_t poolingW = this->poolingW;
    uint32_t strideH = this->strideH;
    uint32_t strideW = this->strideW;
    uint32_t paddingH = this->paddingH;
    uint32_t paddingW = this->paddingW;
    uint32_t batch_size = inputs.size();

    if (!strideH || !strideW) {
        LOG(ERROR) << "stride set error";
        return InferStatus::rInferFailedStrideParameterError;
    }

    for (uint32_t i = 0; i < batch_size; i++){ 
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        if (input == nullptr || input->empty()) {
            LOG(ERROR) << "Input of maxpooling is empty";
            return InferStatus::rInferFailedInputEmpty;
        } else {
            uint32_t input_rows = input->rows();
            uint32_t input_cols = input->cols();
            const uint32_t output_rows = (uint32_t)std::floor((input_rows - poolingH) / strideH + 1);
            const uint32_t output_cols = (uint32_t)std::floor((input_cols - poolingW) / strideW + 1);
            if (!output_rows || !output_cols) {
                LOG(ERROR) << "The output size of max pooling layer is less than zero";
                return InferStatus::rInferFailedOutputSizeError;
            } else {
                std::shared_ptr<Tensor<T>>& output = outputs[i];
                if (output == nullptr || output->empty()) continue;
                if (output->rows() != output_rows || output->cols() != output_cols) {
                    LOG(ERROR) << "The output size of max pooling layer is not adapting";
                    return InferStatus::rInferFailedOutputSizeError;
                }
            }
        }
    }
    #pragma omp parallel for num_threads(batch_size)
    for (uint32_t i = 0; i < batch_size; i++){
        const std::shared_ptr<Tensor<T>>& input_data = inputs.at(i)->clone();
        input_data->padding({paddingH, paddingH, paddingW, paddingW}, std::numeric_limits<T>::lowest());
        const uint32_t input_rows = input_data->rows();
        const uint32_t input_cols = input_data->cols();
        const uint32_t input_channels = input_data->channels();
        const uint32_t output_channels = input_channels;
        const uint32_t output_rows = (uint32_t)std::floor((input_rows - poolingH) / strideH + 1);
        const uint32_t output_cols = (uint32_t)std::floor((input_cols - poolingW) / strideW + 1);
        std::shared_ptr<Tensor<T>> output = outputs[i];
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<T>>(output_rows, output_cols, output_channels);
            outputs[i] = output;
        }
        CHECK(output_rows == output->rows() && output_cols == output->cols() && output_channels == output->channels())
             << "The output size of max pooling layer is error";

        for (uint32_t ch = 0; ch < output_channels; ++ch) {
            const arma::Mat<T>& in_channel = input_data->at(ch);
            arma::Mat<T>& out_channel = output->at(ch);
            uint32_t up_row = 0, left_col = 0, down_row = poolingH - 1, right_col = poolingW - 1;
            for (uint32_t row = 0; row < output_rows; ++row) {
                for (uint32_t col = 0; col < output_cols; ++col) {
                    const arma::Mat<T>& subM = in_channel.submat(up_row + row * strideH, left_col + col * strideW, 
                                                                 down_row + row * strideH, right_col + col * strideW);
                    out_channel.at(row, col) = subM.max();
                
                }
            }
        }
    }   
    return InferStatus::rInferSuccess;                                
}

template<class T>
ParseParamAttrStatus rcMaxPoolingLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                           std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr) << "null operator";
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = op->params;

    if (params.find("stride") == params.end()) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParamAttrStatus::rParameterMissingStride;
    }
    const std::shared_ptr<RuntimeParamIntArray> stride = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["stride"]);
    if (stride == nullptr) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParamAttrStatus::rParameterMissingStride;
    }

    if (params.find("padding") == params.end()) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }
    const std::shared_ptr<RuntimeParamIntArray> padding = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["padding"]);
    if (padding == nullptr) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }

    if (params.find("kernel_size") == params.end()) {
        LOG(ERROR) << "Can not find the kernel_size parameter";
        return ParseParamAttrStatus::rParameterMissingKernel;
    }
    const std::shared_ptr<RuntimeParamIntArray> kernel_size = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["kernel_size"]);
    if (kernel_size == nullptr) {
        LOG(ERROR) << "Can not find the kernel_size parameter";
        return ParseParamAttrStatus::rParameterMissingKernel;
    }

    auto &stride_vals = stride->value;
    auto &padding_vals = padding->value;
    auto &pooling_vals = kernel_size->value;

    const uint32_t dims = 2;
    if (padding_vals.size() != dims) {
        LOG(ERROR) << "Can not find the right padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }

    if (stride_vals.size() != dims) {
        LOG(ERROR) << "Can not find the right stride parameter";
        return ParseParamAttrStatus::rParameterMissingStride;
    }

    if (pooling_vals.size() != dims) {
        LOG(ERROR) << "Can not find the right kernel size parameter";
        return ParseParamAttrStatus::rParameterMissingKernel;
    }

    layer = std::make_shared<rcMaxPoolingLayer<T>>(pooling_vals[0], pooling_vals[1],
                                                        stride_vals[0], stride_vals[1],
                                                        padding_vals[0], padding_vals[1]);
    return ParseParamAttrStatus::rParameterAttrParseSuccess;

}

INSTALLCLASS(rcMaxPoolingLayer);
RCREGISTER_CREATOR(maxpooling, "nn.MaxPool2d", rcMaxPoolingLayer);

}