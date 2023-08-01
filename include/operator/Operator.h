#ifndef OPERATOR_H_
#define OPERATOR_H_

#include <iostream>

namespace rq {

enum class OperatorType {
    rOperatorUnknown = -1,
    rOperatorRelu = 0,
    rOperatorSigmoid = 1,
    rOperatorMaxPooling = 2,
    rOperatorExpression = 3,
    rOperatorConv = 4
};

class Operator {
public:
    OperatorType opType = OperatorType::rOperatorUnknown;

    virtual ~Operator() = default;

    explicit Operator(OperatorType opType) : opType(opType){};
};


}
#endif