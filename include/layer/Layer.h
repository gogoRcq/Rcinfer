#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <data/Tensor.h>
#include <glog/logging.h>

namespace rq {

template<class T>
class Layer {
private:
    std::string layerName;
public:
    Layer(std::string layerName) : layerName(layerName) {};

    virtual ~Layer() = default;

    virtual void forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs);
};


}


#endif