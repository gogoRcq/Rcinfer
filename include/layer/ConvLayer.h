#include "Layer.h"
#include "LayerRegister.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "operator/ConvOperator.h"
#include "operator/Operator.h"
#include <memory>

namespace rq {

template<class T>
class ConvLayer : public Layer<T> {
public:
    ConvLayer(const std::shared_ptr<Operator>& op) : Layer<T>("conv") {
        CHECK(op != nullptr && op->opType == OperatorType::rOperatorConv);
        ConvOperator<T> *convOP = dynamic_cast<ConvOperator<T>*>(op.get());
        CHECK(convOP != nullptr);
        this->op = std::make_unique<ConvOperator<T>>(*convOP);
    }

    ~ConvLayer() override = default;
    
    virtual void forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                          std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;
    
    static std::shared_ptr<Layer<T>> creatorInstance(const std::shared_ptr<Operator>& op);


private:
    std::unique_ptr<ConvOperator<T>> op;
};

}