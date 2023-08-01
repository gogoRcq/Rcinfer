#ifndef CONVOPERATOR_H_
#define CONVOPERATOR_H_

#include "Operator.h"
#include "common.h"
#include "data/Tensor.h"
#include <_types/_uint32_t.h>
#include <memory>
#include <vector>

namespace rq {

template<class T>
class ConvOperator : public Operator {
public:
    uint32_t getStrideH() const;    
    uint32_t getStrideW() const;
    uint32_t getPaddingH() const;
    uint32_t getPaddingW() const;
    uint32_t getGroups() const;
    bool isUseBias() const;
    const std::vector<std::shared_ptr<Tensor<T>>>& weights() const;
    const std::vector<std::shared_ptr<Tensor<T>>>& bias() const;

    void setStrideH(uint32_t strideH);
    void setStrideW(uint32_t strideW);
    void setPaddingH(uint32_t paddingH);
    void setPaddingW(uint32_t paddingW);
    void setGroups(uint32_t groups);
    void setIsUseBias(bool useBias);
    void setWeights(std::vector<std::shared_ptr<Tensor<T>>>& weights);
    void setBias(std::vector<std::shared_ptr<Tensor<T>>>& bias);

    ConvOperator(bool useBias, uint32_t groups, uint32_t strideH, uint32_t strideW, uint32_t paddingH, uint32_t paddingW)
                : Operator(OperatorType::rOperatorConv), useBias(useBias), strideH(strideH), strideW(strideW), paddingH(paddingH), paddingW(paddingW), groups(groups) {};

private:
    bool useBias = false;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t paddingH = 0;
    uint32_t paddingW = 0;
    uint32_t groups = 1;
    std::vector<std::shared_ptr<Tensor<T>>> weights_;
    std::vector<std::shared_ptr<Tensor<T>>> bias_;
};

}

#endif