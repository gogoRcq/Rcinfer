#ifndef ReluLAYER_H_
#define ReluLAYER_H_

#include "Layer.h"
#include "operator/ReluOperator.h"
#include "LayerRegister.h"
#include <memory>

namespace rq {

template<class T>
class ReluLayer : public Layer <T> {
private:
    std::unique_ptr<ReluOperator<T>> op;
public:
    ReluLayer(const std::shared_ptr<Operator> &op) : Layer<T>("Relu") {
        CHECK(op->opType == OperatorType::rOperatorRelu);
        ReluOperator<T> *reluOp = dynamic_cast<ReluOperator<T> *>(op.get());
        CHECK(reluOp != nullptr);
        this->op = std::make_unique<ReluOperator<T>>(reluOp->getThresh());
    };

    ~ReluLayer() override = default;

    virtual void forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static std::shared_ptr<Layer<T>> creatorInstance(const std::shared_ptr<Operator> &op);
};



}
#endif