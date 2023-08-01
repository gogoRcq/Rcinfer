#include "operator/ConvOperator.h"
#include "LayerRegister.h"
#include <_types/_uint32_t.h>

namespace rq {

template<class T>
void ConvOperator<T>::setBias(std::vector<std::shared_ptr<Tensor<T>>> &bias) {
    this->bias_ = bias;
}

template<class T>
void ConvOperator<T>::setWeights(std::vector<std::shared_ptr<Tensor<T>>> &weights) {
    this->weights_ = weights;
}

template<class T>
void ConvOperator<T>::setPaddingH(uint32_t paddingH) {
    this->paddingH = paddingH;
}

template<class T>
void ConvOperator<T>::setPaddingW(uint32_t paddingW) {
    this->paddingW = paddingW;
}

template<class T>
void ConvOperator<T>::setStrideH(uint32_t strideH) {
    this->strideH = strideH;
}

template<class T>
void ConvOperator<T>::setStrideW(uint32_t strideW) {
    this->strideW = strideW;
}

template<class T>
void ConvOperator<T>::setGroups(uint32_t groups) {
    this->groups = groups;
}

template<class T>
void ConvOperator<T>::setIsUseBias(bool useBias) {
    this->useBias = useBias;
}

template<class T>
const std::vector<std::shared_ptr<Tensor<T>>>& ConvOperator<T>::bias() const {
    return this->bias_;
}

template<class T>
const std::vector<std::shared_ptr<Tensor<T>>>& ConvOperator<T>::weights() const {
    return this->weights_;
}

template<class T>
uint32_t ConvOperator<T>::getPaddingH() const {
    return this->paddingH; 
}

template<class T>
uint32_t ConvOperator<T>::getPaddingW() const {
    return this->paddingW;
}

template<class T>
uint32_t ConvOperator<T>::getStrideH() const {
    return this->strideH;
}

template<class T>
uint32_t ConvOperator<T>::getStrideW() const {
    return this->strideW;
}

template<class T>
uint32_t ConvOperator<T>::getGroups() const {
    return this->groups;
}

template<class T>
bool ConvOperator<T>::isUseBias() const {
    return this->useBias;
}

INSTALLCLASS(ConvOperator);

}