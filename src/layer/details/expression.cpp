#include "details/expression.h"
#include "glog/logging.h"
#include "runtime/RuntimeParam.h"
#include "runtime/StateCode.h"
#include <memory>
#include <stack>
#include "common.h"
#include "layer/abstract/rcLayerRegister.h"

namespace rq {

template<class T>
InferStatus rcExpressionLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                           std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "The input feature map of expression layer is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (outputs.empty()) {
        LOG(ERROR) << "Output and input size of the expression layer is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    CHECK(this->expressionParser != nullptr) << "null expr parser";
    this->expressionParser->tokenizer(false);
    const std::vector<Token>& tokens = this->expressionParser->tokens(); 
    CHECK(!tokens.empty()) << "The expression in the expression layer tokenize failed!";

    for (int i = 0; i < inputs.size(); ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        if (!input || input->empty()) {
            LOG(ERROR) << "The output of the expression layer is empty";
            return InferStatus::rInferFailedInputOutSizeAdaptingError;
        }
    }
    uint32_t batch_size = outputs.size();
    for (uint32_t i = 0; i < batch_size; ++i) {
        if (outputs.at(i) == nullptr || outputs.at(i)->empty()) {
            LOG(ERROR) << "The output of the expression layer is empty";
            return InferStatus::rInferFailedInputOutSizeAdaptingError;
        }
        outputs.at(i)->fill((T)0);
    }
    std::stack<std::vector<std::shared_ptr<Tensor<T>>>> dataStack;
    const std::vector<std::shared_ptr<TokenNode>> tokenNodes = expressionParser->generate();
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
        CHECK(outputs[i]->shapes() == data[i]->shapes());
        outputs[i] = data[i];
    }
    return InferStatus::rInferSuccess;        
}

template<class T>
ParseParamAttrStatus rcExpressionLayer<T>::creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                           std::shared_ptr<rcLayer<T>>& layer) {
    CHECK(op != nullptr);
    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>>& params = op->params;

    if (params.find("expr") == params.end()) {
        LOG(ERROR) << "Can not find the expression parameter";
        return ParseParamAttrStatus::rParameterMissingExpr;
    }
    const std::shared_ptr<RuntimeParamString> expr = std::dynamic_pointer_cast<RuntimeParamString>(params["expr"]);
    if (expr == nullptr || expr->type != RuntimeParamType::rParameterString) {
        LOG(ERROR) << "Can not find the expression parameter";
        return ParseParamAttrStatus::rParameterMissingExpr;
    }

    layer = std::make_shared<rcExpressionLayer<T>>(expr->value);
    return ParseParamAttrStatus::rParameterAttrParseSuccess;
}

INSTALLCLASS(rcExpressionLayer);
RCREGISTER_CREATOR(expression, "pnnx.Expression", rcExpressionLayer);

}