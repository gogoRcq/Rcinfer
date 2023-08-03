#ifndef MAXPOOLING_H_
#define MAXPOOLING_H_

#include "layer/abstract/rcLayer.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"

namespace rq {

template<class T>
class rcMaxPoolingLayer : public rcLayer<T> {
public:
    rcMaxPoolingLayer(  uint32_t poolingH, uint32_t poolingW, uint32_t strideH, 
                        uint32_t strideW, uint32_t paddingH, uint32_t paddingW): 
                        rcLayer<T>("maxpooling"), poolingH(poolingH), poolingW(poolingW), 
                        strideH(strideH), strideW(strideW), paddingH(paddingH), paddingW(paddingW) {};
    
    ~rcMaxPoolingLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer);

private:
    uint32_t poolingH;
    uint32_t poolingW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
};

}

#endif