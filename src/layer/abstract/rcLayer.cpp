#include "layer/abstract/rcLayer.h"
#include "common.h"
#include "glog/logging.h"

namespace rq {

template<class T>
void rcLayer<T>::setBias(const std::vector<std::shared_ptr<Tensor<T>>>& bias) {
    LOG(FATAL) << this->layerName << "this layer is not implement yet";
}

template<class T>
void rcLayer<T>::setBias(const std::vector<T>& bias) {
    LOG(FATAL) << this->layerName << "this layer is not implement yet";
}

template<class T>
void rcLayer<T>::setWights(const std::vector<std::shared_ptr<Tensor<T>>>& weights) {
    LOG(FATAL) << this->layerName << "this layer is not implement yet";
}

template<class T>
void rcLayer<T>::setWights(const std::vector<T>& weights) {
    LOG(FATAL) << this->layerName << "this layer is not implement yet";
}

template<class T>
const std::vector<std::shared_ptr<Tensor<T>>>& rcLayer<T>::getWights() const {
    LOG(FATAL) << this->layerName << "this layer is not implement yet";
}

template<class T>
const std::vector<std::shared_ptr<Tensor<T>>>& rcLayer<T>::getBias() const {
    LOG(FATAL) << this->layerName << "this layer is not implement yet";
}

template<class T>
InferStatus rcLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                     std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    LOG(FATAL) << "this layer is not implement yet";              
}

INSTALLCLASS(rcLayer);

}