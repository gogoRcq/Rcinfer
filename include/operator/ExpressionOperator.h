#ifndef EXPRESSIONOPERATOR_H_
#define EXPRESSIONOPERATOR_H_

#include "Operator.h"
#include "runtime/PraseExpression.h"
#include <string>
#
#include <memory>

namespace rq {

class ExpressionOperator : public Operator {
public:
    explicit ExpressionOperator(const std::string& statement);
    std::vector<std::shared_ptr<TokenNode>> generator();
private:
    std::shared_ptr<ExpressionParser> expressionParser;
    std::vector<std::shared_ptr<TokenNode>> tokenNodes;
};

}

#endif