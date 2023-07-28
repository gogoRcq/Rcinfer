#ifndef MAXPOOLINGOPERATOR_H_
#define MAXPOOLINGOPERATOR_H_

#include "Operator.h"

namespace rq {

class MaxPoolingOperator : public Operator{
private:
    uint32_t poolingH;
    uint32_t poolingW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
public:
    MaxPoolingOperator(uint32_t poolingH, uint32_t poolingW, uint32_t strideH, 
                       uint32_t strideW, uint32_t paddingH, uint32_t paddingW)
    : poolingH(poolingH), poolingW(poolingW), strideH(strideH), 
      strideW(strideW), paddingH(paddingH), paddingW(paddingW), Operator(OperatorType::rOperatorMaxPooling) {};

    ~MaxPoolingOperator() override = default;

    uint32_t getPoolingH() const;
    uint32_t getPoolingW() const;
    uint32_t getStrideH() const;
    uint32_t getStrideW() const;
    uint32_t getPaddingH() const;
    uint32_t getPaddingW() const;

    void setPoolingH(uint32_t poolingH);
    void setPoolingW(uint32_t poolingW);
    void setStrideH(uint32_t strideH);
    void setStrideW(uint32_t strideW);
    void setPaddingH(uint32_t paddingH);
    void setPaddingW(uint32_t paddingW);
};

}




#endif