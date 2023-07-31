#ifndef EXPRESSIONLAYER_H_
#define EXPRESSIONLAYER_H_

#include "Layer.h"
#include "glog/logging.h"
#include "operator/ExpressionOperator.h"
#include "operator/Operator.h"
#include "runtime/PraseExpression.h"
#include <memory>
#include <sys/syslimits.h>

namespace rq {

template<class T>
class ExpressionLayer : public Layer<T> {
public:
    explicit ExpressionLayer(const std::shared_ptr<Operator>& op) : Layer<T>("expression") {
        CHECK(op != nullptr && op->opType == OperatorType::rOperatorExpression) << "error op";
        ExpressionOperator* expressionOp = dynamic_cast<ExpressionOperator*>(op.get());
        CHECK(expressionOp != nullptr);
        this->op = std::make_unique<ExpressionOperator>(*expressionOp);
    }

    ~ExpressionLayer() override = default;

    virtual void forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;
    
    static std::shared_ptr<Layer<T>> creatorInstance(const std::shared_ptr<Operator>& op);
private:
    std::unique_ptr<ExpressionOperator> op;
};

}

#endif