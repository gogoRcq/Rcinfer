#include "details/linear.h"
#include "common.h"
#include "glog/logging.h"
#include "layer/abstract/rcLayerRegister.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <memory>
#include <sys/_types/_int32_t.h>

namespace rq {

template<class T>
InferStatus rcLinearLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
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

    if (this->weights.size() != 1) {
        LOG(ERROR) << "The size of weight parameters is not one";
        return InferStatus::rInferFailedWeightParameterError;
    }

    uint32_t batchSize = inputs.size();
    std::shared_ptr<Tensor<T>>& weight = this->weights.front();
    arma::Mat<T>& weightData = weight->at(0);

    #pragma omp parallel for num_threads(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        CHECK(input != nullptr && !input->empty()) << "null input";
        const std::vector<uint32_t>& input_shapes = input->shapes();
        CHECK(input_shapes.size() == 3 && input_shapes[2] == 1);

        const uint32_t inputRow = input_shapes[0];
        CHECK(weightData.n_rows == outFeatures && weightData.n_cols == inputRow && inFeatures == inputRow);
        const uint32_t inputCol = input_shapes[1];

        arma::Mat<T>& inputData = input->at(0);
        std::shared_ptr<Tensor<T>> output = outputs.at(i);
            if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<T>>(outFeatures, inputCol, 1);
            outputs.at(i) = output;
        }
        CHECK(output->channels() == 1 && output->rows() == outFeatures && output->cols() == inputCol);
        arma::Mat<T>& result = output->at(0);
        result = weightData * inputData;
        if (useBias) {
            CHECK(!this->bias.empty() && this->bias.size() == 1);
            std::shared_ptr<Tensor<T>>& biasData = this->bias[0];
            CHECK(!biasData->empty());
            CHECK(biasData->rows() == outFeatures && biasData->channels() == 1) << "error bias data";
            result += biasData->at(0);
        }
    }
    return InferStatus::rInferSuccess;
}

template<class T>
ParseParamAttrStatus rcLinearLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                       std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr);
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>> params = op->params;

    if (params.find("bias") == params.end()) {
        LOG(ERROR) << "Can not find the bias parameter";
        return ParseParamAttrStatus::rParameterMissingUseBias;
    }
    const std::shared_ptr<RuntimeParamBool> bias = std::dynamic_pointer_cast<RuntimeParamBool>(params["bias"]);
    if (!bias) {
        LOG(ERROR) << "Can not find the bias parameter";
        return ParseParamAttrStatus::rParameterMissingUseBias;
    }

    std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<T>>> attributes = op->attributes;

    if (attributes.find("weight") == attributes.end()) {
        LOG(ERROR) << "can not find weight attribute";
        return ParseParamAttrStatus::rAttrMissingWeight;
    }
    const std::shared_ptr<RuntimeAttr<T>>& weight = attributes["weight"];
    const std::vector<int32_t>& shape = weight->shape;
    if (shape.empty() && shape.size() == 2) {
        LOG(ERROR) << " weight shape is error";
        return ParseParamAttrStatus::rAttrMissingWeight;
    }
    int32_t out_features = shape.at(0);
    int32_t in_features = shape.at(1);
    const std::vector<T> weightVal = weight->get();
    
    layer = std::make_shared<rcLinearLayer<T>>(in_features, out_features, bias->value);

    if (bias->value) {
        if (attributes.find("bias") == attributes.end()) {
            LOG(ERROR) << "can not find bias attribute";
            return ParseParamAttrStatus::rAttrMissingBias;
        }
        const std::shared_ptr<RuntimeAttr<T>>& bias = attributes["bias"];
        const std::vector<T> biasVal = bias->get();
        layer->setBias(biasVal);
    }
    layer->setWeights(weightVal);

    return ParseParamAttrStatus::rParameterAttrParseSuccess;
}

INSTALLCLASS(rcLinearLayer);
RCREGISTER_CREATOR(linear, "nn.Linear", rcLinearLayer);

}