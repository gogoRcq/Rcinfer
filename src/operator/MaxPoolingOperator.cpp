#include "operator/MaxPoolingOperator.h"

namespace rq {

uint32_t MaxPoolingOperator::getPoolingH() const {
    return this->poolingH;
}

uint32_t MaxPoolingOperator::getPoolingW() const {
    return this->poolingW;
}

uint32_t MaxPoolingOperator::getStrideH() const {
    return this->strideH;
}

uint32_t MaxPoolingOperator::getStrideW() const {
    return this->strideW;
}

uint32_t MaxPoolingOperator::getPaddingH() const {
    return this->paddingH;
}

uint32_t MaxPoolingOperator::getPaddingW() const {
    return this->paddingW;
}

void MaxPoolingOperator::setPoolingH(uint32_t poolingH) {
    this->poolingH = poolingH;
}

void MaxPoolingOperator::setPoolingW(uint32_t poolingW) {
    this->poolingW = poolingW;
}

void MaxPoolingOperator::setStrideH(uint32_t strideH) {
    this->strideH = strideH;
}

void MaxPoolingOperator::setStrideW(uint32_t strideW) {
    this->strideW = strideW;
}

void MaxPoolingOperator::setPaddingH(uint32_t paddingH) {
    this->paddingH = paddingH;
}

void MaxPoolingOperator::setPaddingW(uint32_t paddingW) {
    this->paddingW = paddingW;
}

}
