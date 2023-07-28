#ifndef SIGMOIDOPERATOR_H_
#define SIGMOIDOPERATOR_H_

#include "Operator.h"

namespace rq {

class SigmoidOperator : public Operator {
private:
    /* data */
public:
    SigmoidOperator();
    ~SigmoidOperator() override = default;
};


}

#endif