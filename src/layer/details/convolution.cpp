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