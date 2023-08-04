#ifndef ADAPTIVEAVERAGEPOOLING_H_
#define ADAPTIVEAVERAGEPOOLING_H_

#include "common.h"
#include "layer/abstract/rcLayer.h"
#include "layer/abstract/rcLayerRegister.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>

namespace rq {

template<class T>
class AdaptiveAveragePoolingLayer : public rcLayer<T> {
public:
    AdaptiveAveragePoolingLayer(uint32_t outputH, uint32_t outputW) : rcLayer<T>("AdaptiveAveragePoolingr"),
                                                                      outputH(outputH), outputW(outputW) {};
    
    ~AdaptiveAveragePoolingLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer); 

private:
    uint32_t outputH;
    uint32_t outputW;
};


}

#endif