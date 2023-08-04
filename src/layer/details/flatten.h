#ifndef FLATTEN_H_
#define FLATTEN_H_

#include "layer/abstract/rcLayer.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"
#include <sys/_types/_int32_t.h>

namespace rq {

template<class T>
class rcFlattenLayer : public rcLayer<T> {
public:
    rcFlattenLayer(int32_t startDim, int32_t endDim) : rcLayer<T>("flatten"), startDim(startDim), endDim(endDim){};

    ~rcFlattenLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer);   

private:
    int32_t startDim = 1;
    int32_t endDim = -1;
};

}

#endif