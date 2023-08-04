#include "details/convolution.h"
#include "common.h"
#include "glog/logging.h"
#include "layer/abstract/rcLayerRegister.h"
#include "runtime/RuntimeParam.h"
#include "runtime/StateCode.h"
#include <memory>
#include <vector>

namespace rq {

template<class T>
InferStatus rcConvolutionLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                            std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input of conv is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    if (this->weights.empty()) {
        LOG(ERROR) << "Weight parameters is empty";
        return InferStatus::rInferFailedWeightParameterError;
    }

    if (this->useBias && this->bias.size() != this->weights.size()) {
        LOG(ERROR) << "The size of the weight and bias is not adapting";
        return InferStatus::rInferFailedBiasParameterError;
    }

    if (this->weights.size() <= 0) {
        LOG(ERROR) << "kernel count must greater than zero";
        return InferStatus::rInferFailedWeightParameterError;
    }

    if (!strideH || !strideW) {
        LOG(ERROR) << "stride set error";
        return InferStatus::rInferFailedStrideParameterError;
    }

    bool useBias = this->useBias;
    const uint32_t paddingH = this->paddingH;
    const uint32_t paddingW = this->paddingW;
    const uint32_t strideH = this->strideH;
    const uint32_t strideW = this->strideW;
    const uint32_t groups = this->groups;
    const uint32_t batchSize = inputs.size();

    #pragma omp parallel for num_threads(batchSize)
    for (uint32_t i = 0; i < batchSize; ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        CHECK(input != nullptr && !input->empty());

        std::shared_ptr<Tensor<T>> input_;
        if (paddingH > 0 || paddingW > 0) {
            input_ = input->clone();
            input_->padding({paddingH, paddingH, paddingW, paddingW}, (T)0.0f);
        } else {
            input_ = input;
        }
        const uint32_t input_channels = input_->channels();
        const uint32_t input_rows = input_->rows();
        const uint32_t input_cols = input_->cols();
        const uint32_t output_channels = input_channels;
        const uint32_t kernel_counts = this->weights.size();
        const uint32_t kernel_rows = this->weights[0]->rows();
        const uint32_t kernel_cols = this->weights[0]->cols();
        CHECK(kernel_cols > 0 && kernel_rows > 0);
        const uint32_t output_rows = (uint32_t)std::floor((input_rows - kernel_rows) / strideH + 1);
        const uint32_t output_cols = (uint32_t)std::floor((input_cols - kernel_cols) / strideW + 1);
        CHECK(output_rows > 0 && output_cols > 0) << "The size of the output feature map is less than zero";
        if (groups != 1) {
            CHECK(kernel_counts % groups == 0);
            CHECK(input_channels % groups == 0);
        }
        for (const auto& weight : this->weights) {
            CHECK(weight->rows() == kernel_rows);
            CHECK(weight->cols() == kernel_cols);
            CHECK(weight->channels() == input_channels / groups);
        }
        uint32_t row_len = kernel_cols * kernel_cols; // kernel展开的长度
        uint32_t col_len = output_rows * output_cols; // 一行的长度，其实也就是一个结果的一个 channel
        uint32_t input_group_channels = input_channels / groups; // 每组的input的 channel 数
        uint32_t kernel_count_group = kernel_counts / groups; // 每组有多少个 kernel
        
        for (int g = 0; g < groups; ++g) {
            std::vector<arma::Mat<T>> kernel_mat_arr(kernel_count_group);
            arma::Mat<T> temp(1, row_len *  input_group_channels);
            for (int k = 0; k < kernel_count_group; ++k) {
                const std::shared_ptr<Tensor<T>>& kernel = this->weights[k + g * kernel_count_group];
                //std::memcpy(temp.memptr(), kernel->data().memptr(), sizeof(T) * row_len * input_group_channels);
                for (int igc = 0; igc < input_group_channels; ++igc) {
                    std::memcpy(temp.memptr() + row_len * igc, kernel->at(igc).memptr(), row_len * sizeof(T));
                }
                kernel_mat_arr[k] = temp;
            }
            arma::Mat<T> input_mat(row_len * input_group_channels, col_len);
            for (int igc = 0; igc < input_group_channels; ++igc) {
                const arma::Mat<T>& channel = input_->at(igc + g * input_group_channels);
                uint32_t cur_col = 0;
                for (uint32_t col = 0; col < output_cols; ++col) {
                    for (uint32_t row = 0; row < output_rows; ++row) {
                        T* input_mat_ptr = input_mat.colptr(cur_col) + igc * row_len;
                        ++cur_col;
                        for (int kl = 0; kl < kernel_cols; ++kl) {
                            memcpy(input_mat_ptr + kl * kernel_rows, channel.colptr(kl + col * strideW) + row * strideH, sizeof(T) * kernel_rows);
                        }
                    }
                }
            }

            std::shared_ptr<Tensor<T>> output = outputs[i];
            if (output == nullptr || output->empty()) {
                output = std::make_shared<Tensor<T>>(output_rows, output_cols, kernel_counts);
                outputs[i] = output;
            }

            std::vector<arma::Mat<T>> output_mats(kernel_count_group);
            for (int om = 0; om < kernel_count_group; ++om) {
                output_mats[om] = kernel_mat_arr[om] * input_mat;
            }
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < kernel_count_group; ++k) {
                std::shared_ptr<Tensor<T>> bias_;
                if (!this->bias.empty() && useBias) {
                    bias_ = this->bias[k];
                }
                arma::Mat<T> output_mat = output_mats[k];
                CHECK(output_mat.size() == output_cols * output_rows);
                output_mat.reshape(output_rows, output_cols);
                if (!this->bias.empty() && useBias) {
                    T bias_val = bias_->index(0);
                    output_mat += bias_val;
                }
                output->at(k + g * kernel_count_group) = std::move(output_mat);
            }
        }
    }

    return InferStatus::rInferSuccess;                    
}

template<class T>
ParseParamAttrStatus rcConvolutionLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                            std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr);
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = op->params;
    std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<T>>> attributes = op->attributes;

    if (params.find("dilation") == params.end()) {
        LOG(ERROR) << "Can not find the dilation parameter";
        return ParseParamAttrStatus::rParameterMissingDim;
    }
    const std::shared_ptr<RuntimeParamIntArray> dilation = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["dilation"]);
    if (dilation == nullptr || dilation->value.size() != 2) {
        LOG(ERROR) << "Can not find the dilation parameter";
        return ParseParamAttrStatus::rParameterMissingDim;
    }
    CHECK(dilation->value[0] == 1 && dilation->value[1] == 1) << "Only support dilation value equals to one!";

    if (params.find("in_channels") == params.end()) {
        LOG(ERROR) << "Can not find in_channels parameter";
        return ParseParamAttrStatus::rParameterMissingInChannel;
    }
    const std::shared_ptr<RuntimeParamInt> in_channel = std::dynamic_pointer_cast<RuntimeParamInt>(params["in_channels"]);
    if (!in_channel) {
        LOG(ERROR) << "Can not find in_channels parameter";
        return ParseParamAttrStatus::rParameterMissingInChannel;
    }

    if (params.find("out_channels") == params.end()) {
        LOG(ERROR) << "Can not find out_channels parameter";
        return ParseParamAttrStatus::rParameterMissingOutChannel;
    }
    const std::shared_ptr<RuntimeParamInt> out_channel = std::dynamic_pointer_cast<RuntimeParamInt>(params["out_channels"]);
    if (!out_channel) {
        LOG(ERROR) << "Can not find out_channels parameter";
        return ParseParamAttrStatus::rParameterMissingOutChannel;
    }

    if (params.find("padding") == params.end()) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }
    const std::shared_ptr<RuntimeParamIntArray> padding = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["padding"]);
    if (!padding) {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }

    if (params.find("bias") == params.end()) {
        LOG(ERROR) << "Can not find the bias parameter";
        return ParseParamAttrStatus::rParameterMissingUseBias;
    }
    const std::shared_ptr<RuntimeParamBool> bias = std::dynamic_pointer_cast<RuntimeParamBool>(params["bias"]);
    if (!bias) {
        LOG(ERROR) << "Can not find the bias parameter";
        return ParseParamAttrStatus::rParameterMissingUseBias;
    }

    if (params.find("stride") == params.end()) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParamAttrStatus::rParameterMissingStride;
    }
    const std::shared_ptr<RuntimeParamIntArray> stride = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["stride"]);
    if (!stride) {
        LOG(ERROR) << "Can not find the stride parameter";
        return ParseParamAttrStatus::rParameterMissingStride;
    }

    if (params.find("kernel_size") == params.end()) {
        LOG(ERROR) << "Can not find the kernel parameter";
        return ParseParamAttrStatus::rParameterMissingKernel;
    }
    const std::shared_ptr<RuntimeParamIntArray> kernel_size = std::dynamic_pointer_cast<RuntimeParamIntArray>(params["kernel_size"]);
    if (!kernel_size) {
        LOG(ERROR) << "Can not find the kernel parameter";
        return ParseParamAttrStatus::rParameterMissingKernel;
    }

    if (params.find("padding_mode") != params.end()) {
        const std::shared_ptr<RuntimeParamString> padding_mode = std::dynamic_pointer_cast<RuntimeParamString>(params["padding_mode"]);
        if (padding_mode == nullptr) {
            LOG(ERROR) << "Can not find the padding parameter";
            return ParseParamAttrStatus::rParameterMissingPadding;
        } else {
            const std::string& padding_mode_str = padding_mode->value;
            if (padding_mode_str != "zeros") {
                LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
                return ParseParamAttrStatus::rParameterMissingPadding;
            }
        }
    } else {
        LOG(ERROR) << "Can not find the padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }

    if (params.find("groups") == params.end()) {
        LOG(ERROR) << "Can not find the groups parameter";
        return ParseParamAttrStatus::rParameterMissingGroups;
    }
    const std::shared_ptr<RuntimeParamInt> groups = std::dynamic_pointer_cast<RuntimeParamInt>(params["groups"]);
    if (!groups) {
        LOG(ERROR) << "Can not find the groups parameter";
        return ParseParamAttrStatus::rParameterMissingGroups;
    }

    const uint32_t dims = 2;
    const std::vector<int>& kernels = kernel_size->value;
    const std::vector<int>& paddings = padding->value;
    const std::vector<int>& strides = stride->value;
    if (paddings.size() != dims) {
        LOG(ERROR) << "Can not find the right padding parameter";
        return ParseParamAttrStatus::rParameterMissingPadding;
    }

    if (strides.size() != dims) {
        LOG(ERROR) << "Can not find the right stride parameter";
        return ParseParamAttrStatus::rParameterMissingStride;
    }

    if (kernels.size() != dims) {
        LOG(ERROR) << "Can not find the right kernel size parameter";
        return ParseParamAttrStatus::rParameterMissingKernel;
    }

    layer = std::make_shared<rcConvolutionLayer<T>>(bias->value, out_channel->value, in_channel->value, groups->value,
                                                    kernels[0], kernels[1], strides[0], strides[1], paddings[0], paddings[1]);
    
    // load weight and bias
    if (bias->value) {
        if (attributes.find("bias") == attributes.end()) {
            LOG(ERROR) << "can not find bias attribute";
            return ParseParamAttrStatus::rAttrMissingBias;
        }
        const std::shared_ptr<RuntimeAttr<T>>& bias = attributes["bias"];
        const std::vector<int32_t>& shape = bias->shape;
        if (shape.empty() || shape[0] != out_channel->value) {
            LOG(ERROR) << " bias shape is error in conv";
            return ParseParamAttrStatus::rAttrMissingBias;
        }
        const std::vector<T>& biasVal = bias->get();
        layer->setBias(biasVal);
    }

    if (attributes.find("weight") == attributes.end()) {
        LOG(ERROR) << "can not find weight attribute";
        return ParseParamAttrStatus::rAttrMissingWeight;
    }
    const std::shared_ptr<RuntimeAttr<T>>& weight = attributes["weight"];
    const std::vector<int32_t>& shape = weight->shape;
    if (shape.empty()) {
        LOG(ERROR) << " weight shape is error in conv";
        return ParseParamAttrStatus::rAttrMissingWeight;
    }
    const std::vector<T>& weightVal = weight->get();
    layer->setWeights(weightVal);
    return ParseParamAttrStatus::rParameterAttrParseSuccess;
}

INSTALLCLASS(rcConvolutionLayer);
RCREGISTER_CREATOR(conv, "nn.Conv2d", rcConvolutionLayer);

}