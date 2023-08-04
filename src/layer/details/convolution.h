#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include "layer/abstract/rcParamLayer.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <sys/types.h>

namespace rq {

template<class T>
class rcConvolutionLayer : public rcParamLayer<T> {
public:
    rcConvolutionLayer(bool useBias, uint32_t outputChannel, uint32_t inputChannel, uint32_t groups, uint32_t kernelH, 
                       uint32_t kernelW, uint32_t strideH, uint32_t strideW, uint32_t paddingH, uint32_t paddingW):
                       rcParamLayer<T>("conv"), useBias(useBias), strideH(strideH), 
                       strideW(strideW), paddingH(paddingH), paddingW(paddingW), groups(groups) {
        if (groups != 1) {
            inputChannel /= groups;
        }
        if (useBias) this->initBias(outputChannel, 1, 1, 1);
        this->initWights(outputChannel, kernelH, kernelW, inputChannel);
    };
    
    ~rcConvolutionLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer);

private:
    bool useBias = false;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t paddingH = 0;
    uint32_t paddingW = 0;
    uint32_t groups = 1;
};

}

#endif