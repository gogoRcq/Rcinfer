#include "layer/ExpressionLayer.h"
#include "common.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "layer/Layer.h"
#include "operator/Operator.h"
#include "runtime/PraseExpression.h"
#include <_types/_uint32_t.h>
#include <memory>
#include <stack>
#include <sys/_types/_int32_t.h>
#include "common.h"
#include "LayerRegister.h"

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

    uint32_t batch_size = outputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(outputs[i] != nullptr && !outputs[i]->empty());
        outputs[i]->fill((T)0.0f);
    }
    CHECK(this->op != nullptr && this->op->opType == OperatorType::rOperatorExpression);
    std::stack<std::vector<std::shared_ptr<Tensor<T>>>> dataStack;
    const std::vector<std::shared_ptr<TokenNode>> tokenNodes = this->op->generator();
    for (const auto& tokenNode : tokenNodes) {
        if (tokenNode->numIndex >= 0) {
            auto startIdex = tokenNode->numIndex * batch_size;
            std::vector<std::shared_ptr<Tensor<T>>> temp;
            for (int i = 0; i < batch_size; ++i) {
                CHECK(i + startIdex < inputs.size()) << "error inputs";
                temp.emplace_back(inputs[i + startIdex]);
            }
            dataStack.push(temp);
        } else {
            const int32_t op = tokenNode->numIndex;
            CHECK(dataStack.size() >= 2) << "error stack";

            std::vector<std::shared_ptr<Tensor<T>>> data1 = dataStack.top();
            CHECK(data1.size() == batch_size) << "error inputs";
            dataStack.pop();

            std::vector<std::shared_ptr<Tensor<T>>> data2 = dataStack.top();
            CHECK(data2.size() == batch_size) << "error inputs";
            dataStack.pop();

            std::vector<std::shared_ptr<Tensor<T>>> temp(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                if (-op == int32_t(TokenType::TokenAdd)) {
                    temp[i] = Tensor<T>::ElementAdd(data1[i], data2[i]);
                } else if (-op == int32_t(TokenType::TokenMul)) {
                    temp[i] = Tensor<T>::ElementMultiply(data1[i], data2[i]);
                } else {
                    LOG(FATAL) << "Unknown operator type: " << op;
                }
            }
            dataStack.push(temp);
        }
    }  
    CHECK(dataStack.size() == 1) << "calculate error";
    std::vector<std::shared_ptr<Tensor<T>>> data = dataStack.top();
    CHECK(data.size() == batch_size) << "error outs";
    dataStack.pop();
    for (int i = 0; i < batch_size; ++i) {
        outputs[i] = data[i];
    }
}

INSTALLCLASS(ExpressionLayer);
REGISTER_CREATOR(expression, OperatorType::rOperatorExpression, ExpressionLayer);

}