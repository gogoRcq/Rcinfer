#ifndef RELU_H_
#define RELU_H_

#include "layer/abstract/rcLayer.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"
namespace rq {

template<class T>
class rcReluLayer : public rcLayer<T> {
public:
    rcReluLayer() : rcLayer<T>("Relu") {};

    ~rcReluLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer);
};

}

#endif