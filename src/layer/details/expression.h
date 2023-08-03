#ifndef EXPRESSION_H_
#define EXPRESSION_H_

#include "layer/abstract/rcLayer.h"
#include "runtime/PraseExpression.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"
#include <memory>
#include <string>
#include <type_traits>

namespace rq {

template<class T>
class rcExpressionLayer : public rcLayer<T> {
public:
    rcExpressionLayer(const std::string& statement) : rcLayer<T>("expression"), 
                      expressionParser(std::make_unique<ExpressionParser>(statement)){};

    ~rcExpressionLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer);

private:
    std::unique_ptr<ExpressionParser> expressionParser;
};

}

#endif