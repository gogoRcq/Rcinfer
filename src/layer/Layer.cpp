#include "layer/Layer.h"
#include "common.h"

namespace rq {

template<class T>
void Layer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                    std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    LOG(FATAL) << "The layer " << this->layerName << " not implement yet!";  
}

INSTALLCLASS(Layer);

} 
