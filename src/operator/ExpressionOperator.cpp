#include "operator/ExpressionOperator.h"
#include "operator/Operator.h"
#include "runtime/PraseExpression.h"
#include <algorithm>
#include <memory>
#include "glog/logging.h"

namespace rq {

ExpressionOperator::ExpressionOperator(const std::string& statement) : Operator(OperatorType::rOperatorExpression) ,
                                       expressionParser(std::make_shared<ExpressionParser>(statement)) {

}

std::vector<std::shared_ptr<TokenNode>> ExpressionOperator::generator() {
    CHECK(this->expressionParser != nullptr);
    this->tokenNodes = this->expressionParser->generate();
    return this->tokenNodes;
}

}