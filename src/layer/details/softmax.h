#ifndef SOFTMAX_H_
#define SOFTMAX_H_

#include "layer/abstract/rcLayer.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"

namespace rq {

template<class T>
class rcSoftmaxLayer : public rcLayer<T> {
public:
    rcSoftmaxLayer() : rcLayer<T>("softmax"){};

    ~rcSoftmaxLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

};

}

#endif