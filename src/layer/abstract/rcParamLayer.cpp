#include "layer/abstract/rcParamLayer.h"
#include "common.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include <_types/_uint32_t.h>
#include <memory>
#include <vector>

namespace rq {

template<class T>
void rcParamLayer<T>::setBias(const std::vector<std::shared_ptr<Tensor<T>>>& bias) {
    if (!this->bias.empty()) {
        CHECK(bias.size() == this->bias.size());
        for (int i = 0; i < bias.size(); ++i) {
            CHECK(this->bias[i] != nullptr);
            CHECK(this->bias[i]->rows() == bias[i]->rows());
            CHECK(this->bias[i]->cols() == bias[i]->cols());
            CHECK(this->bias[i]->channels() == bias[i]->channels());
        }
    }
    this->bias = bias;
}

template<class T>
void rcParamLayer<T>::setWeights(const std::vector<std::shared_ptr<Tensor<T>>>& weights) {
    if (!this->weights.empty()) {
        CHECK(this->weights.size() == weights.size());
        for (int i = 0; i < weights.size(); ++i) {
            CHECK(this->weights[i] != nullptr);
            CHECK(this->weights[i]->rows() == weights[i]->rows());
            CHECK(this->weights[i]->cols() == weights[i]->cols());
            CHECK(this->weights[i]->channels() == weights[i]->channels());
        }
    }
    this->weights = weights;
}

template<class T>
void rcParamLayer<T>::setBias(const std::vector<T>& bias) {
    const uint32_t elementSize = bias.size();
    uint32_t biasSize = 0;
    const uint32_t batchSize = this->bias.size();
    for (uint32_t i = 0; i < batchSize; ++i) {
        biasSize += this->bias[i]->size();
    }

    CHECK(biasSize == elementSize);
    CHECK(elementSize % batchSize == 0);
    const uint32_t blobSize = elementSize / batchSize;
    for (uint32_t i = 0; i < batchSize; ++i) {
        const uint32_t startOffset = i * blobSize;
        const uint32_t endOffset = startOffset + blobSize;
        this->bias[i]->fill(std::vector<T>(bias.begin() + startOffset, bias.begin() + endOffset));
    }
}

template<class T>
void rcParamLayer<T>::setWeights(const std::vector<T>& weights) {
    const uint32_t elementSize = weights.size();
    uint32_t weightSize = 0;
    const uint32_t batchSize = this->weights.size();
    for (uint32_t i = 0; i < batchSize; ++i) {
        weightSize += this->weights[i]->size();
    }

    CHECK(weightSize == elementSize);
    CHECK(elementSize % batchSize == 0);

    const uint32_t blobSize = elementSize / batchSize;
    for (uint32_t i = 0; i < batchSize; ++i) {
        const uint32_t startOffset = i * blobSize;
        const uint32_t endOffset = startOffset + blobSize;
        this->weights[i]->fill(std::vector<T>(weights.begin() + startOffset, weights.begin() + endOffset));
    }
}

template<class T>
const std::vector<std::shared_ptr<Tensor<T>>>& rcParamLayer<T>::getWights() const {
    return this->weights;
}

template<class T>
const std::vector<std::shared_ptr<Tensor<T>>>& rcParamLayer<T>::getBias() const {
    return this->bias;
}

template<class T>
void rcParamLayer<T>::initBias(uint32_t paramCount, uint32_t paramRow, uint32_t paramCol, uint32_t paramChannel) {
    this->bias = std::vector<std::shared_ptr<Tensor<T>>>(paramCount);
    for (uint32_t i = 0; i < paramCount; ++i) {
        this->bias[i] = std::make_shared<Tensor<T>>(paramRow, paramCol, paramChannel);
    }
}

template<class T>
void rcParamLayer<T>::initWights(uint32_t paramCount, uint32_t paramRow, uint32_t paramCol, uint32_t paramChannel) {
    this->weights = std::vector<std::shared_ptr<Tensor<T>>>(paramCount);
    for (uint32_t i = 0; i < paramCount; ++i) {
        this->weights[i] = std::make_shared<Tensor<T>>(paramRow, paramCol, paramChannel);
    }
}

INSTALLCLASS(rcParamLayer);

}