#ifndef SIGMOIDLAYER_H_
#define SIGMOIDLAYER_H_

#include "Layer.h"
#include "data/Tensor.h"
#include "operator/Operator.h"
#include "operator/SigmoidOperator.h"
#include "LayerRegister.h"

namespace rq {
template<class T>
class SigmoidLayer : public Layer<T> {
private:
    std::unique_ptr<SigmoidOperator> op;
public:
    SigmoidLayer(const std::shared_ptr<Operator>& op) : Layer<T>("sigmoid") {
        CHECK(op->opType == OperatorType::rOperatorSigmoid) << "error optype in sigmoid";
        SigmoidOperator *sgOp = dynamic_cast<SigmoidOperator*>(op.get());
        CHECK(sgOp != nullptr) << "null op";
        this->op = std::make_unique<SigmoidOperator>();
    }

    ~SigmoidLayer() override = default;

    virtual void forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static std::shared_ptr<Layer<T>> creatorInstance(const std::shared_ptr<Operator>& op);
};

}
#endif