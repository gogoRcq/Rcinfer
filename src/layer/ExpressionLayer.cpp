#include "layer/ExpressionLayer.h"
#include "common.h"
#include "glog/logging.h"
#include "layer/Layer.h"
#include "operator/Operator.h"
#include "runtime/PraseExpression.h"
#include <_types/_uint32_t.h>
#include <memory>
#include <stack>

namespace rq {

template<class T>
std::shared_ptr<Layer<T>> ExpressionLayer<T>::creatorInstance(const std::shared_ptr<Operator>& op) {
    return std::make_shared<ExpressionLayer<T>>(op);
}

template<class T>
void ExpressionLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs, 
                                  std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    CHECK(!inputs.empty());
    CHECK(op != nullptr);
    CHECK(op->opType == OperatorType::rOperatorExpression);

    uint32_t batch_size = inputs.size();
    std::stack<std::shared_ptr<TokenNode>> tokenStack;                        
}

INSTALLCLASS(ExpressionLayer);

}