#ifndef ReluOPERATOR_H_
#define ReluOPERATOR_H_

#include "Operator.h"

namespace rq {

template<class T>
class ReluOperator : public Operator {
private:
    T thresh = static_cast<T>(0);

public:
    explicit ReluOperator(T thresh) : thresh(thresh), Operator(OperatorType::rOperatorRelu) {};

    ~ReluOperator () override = default;

    T getThresh() const;

    void setThresh (T val);

};

}

#endif