#ifndef MAXPOOLINGLAYER_H_
#define MAXPOOLINGLAYER_H_

#include "Layer.h"
#include "data/Tensor.h"
#include "operator/MaxPoolingOperator.h"
#include "LayerRegister.h"

namespace rq {

template<class T>
class MaxPoolingLayer : public Layer<T> {
private:
    std::unique_ptr<MaxPoolingOperator> op;
public:
    MaxPoolingLayer(const std::shared_ptr<Operator>& op) : Layer<T>("maxPooling") {
        CHECK(op->opType == OperatorType::rOperatorMaxPooling) << "error optype!";
        MaxPoolingOperator* maxPoolop = dynamic_cast<MaxPoolingOperator*>(op.get());
        CHECK(maxPoolop != nullptr);
        this->op = std::make_unique<MaxPoolingOperator>(*maxPoolop);
    }

    ~MaxPoolingLayer() override = default;

    virtual void forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;
    
    static std::shared_ptr<Layer<T>> creatorInstance(const std::shared_ptr<Operator>& op);
};

}


#endif