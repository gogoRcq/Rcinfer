#include "runtime/PraseExpression.h"
#include <__algorithm/remove_if.h>
#include <_ctype.h>
#include <memory>
#include <string>
#include "glog/logging.h"
#include "gtest/gtest.h"
#include <sys/_types/_int32_t.h>

namespace rq {

const std::vector<Token>& ExpressionParser::tokens() const {
    return this->tokens_;
}

const std::vector<std::string>& ExpressionParser::tokenStrings() const {
    return this->tokenStrings_;
}

void ExpressionParser::ReversePolish(std::shared_ptr<TokenNode>& curNode, std::vector<std::shared_ptr<TokenNode>>& reversePolish){
    if(curNode == nullptr) return;
    ReversePolish(curNode->left, reversePolish);
    ReversePolish(curNode->right, reversePolish);
    reversePolish.emplace_back(curNode);
}

void ExpressionParser::tokenizer(bool needRetoken) {
    if (!needRetoken && !tokens_.empty()) return; // 不需要重新解析

    CHECK(!this->statement.empty()) << "The inputstatement is empty!";
    this->statement.erase(std::remove_if(statement.begin(), statement.end(), [](char c){
        return std::isspace(c);
    }), statement.end());
    CHECK(!this->statement.empty()) << "The inputstatement is empty!";

    for (int32_t i = 0; i < statement.size();) {
        char c = statement.at(i);
        if (c == 'a') {
            CHECK(i + 1 < statement.size() && statement.at(i + 1) == 'd') << "error statement";
            CHECK(i + 2 < statement.size() && statement.at(i + 2) == 'd') << "error statement";
            tokens_.emplace_back(Token(TokenType::TokenAdd, i, i + 3));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + i + 3));
            i += 3;
        } else if (c == 'm') {
            CHECK(i + 1 < statement.size() && statement.at(i + 1) == 'u') << "error statement";
            CHECK(i + 2 < statement.size() && statement.at(i + 2) == 'l') << "error statement";
            tokens_.emplace_back(Token(TokenType::TokenMul, i, i + 3));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + i + 3));
            i += 3;
        } else if (c == 'd') {
            CHECK(i + 1 < statement.size() && statement.at(i + 1) == 'i') << "error statement";
            CHECK(i + 2 < statement.size() && statement.at(i + 2) == 'v') << "error statement";
            tokens_.emplace_back(Token(TokenType::TokenDiv, i, i + 3));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + i + 3));
            i += 3;
        } else if (c == '@') {
            CHECK(i + 1 < statement.size() && std::isdigit(statement.at(i + 1))) << "error statement";
            int j = i + 2;
            while (j < statement.size() && std::isdigit(statement.at(j))) {
                ++j;
            }
            tokens_.emplace_back(Token(TokenType::TokenNumber, i, j));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + j));
            i = j;
        } else if (c == ',') {
            tokens_.emplace_back(Token(TokenType::TokenComma, i, i + 1));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + i + 1));
            ++i;
        } else if (c == '(') {
            tokens_.emplace_back(Token(TokenType::TokenLeftBracket, i, i + 1));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + i + 1));
            ++i;
        } else if (c == ')') {
            tokens_.emplace_back(Token(TokenType::TokenRightBracket, i, i + 1));
            tokenStrings_.emplace_back(std::string(statement.begin() + i, statement.begin() + i + 1));
            ++i;
        } else {
            LOG(FATAL) << "Unknown illegal character: " << c;
        }
    }
}

std::vector<std::shared_ptr<TokenNode>> ExpressionParser::generate() {
    if (tokens_.empty()) {
        tokenizer(true);
    }
    int32_t index = 0;
    auto root = generate_(index);
    CHECK(root != nullptr) << "null tokenNode!";
    CHECK(index == tokens_.size() - 1);
    std::vector<std::shared_ptr<TokenNode>> reversePolish;
    ReversePolish(root, reversePolish);
    return reversePolish;
}

std::shared_ptr<TokenNode> ExpressionParser::generate_(int32_t& index) {
    CHECK(index < tokens_.size());
    const auto token = tokens_.at(index);
    CHECK(token.tokenType == TokenType::TokenAdd || 
          token.tokenType == TokenType::TokenMul || 
          token.tokenType == TokenType::TokenNumber ||
          token.tokenType == TokenType::TokenDiv);
    if (token.tokenType == TokenType::TokenNumber) {
        return std::make_shared<TokenNode>(std::stoi(tokenStrings_.at(index).substr(1)), nullptr, nullptr);
    } else if (token.tokenType == TokenType::TokenMul || token.tokenType == TokenType::TokenAdd || token.tokenType == TokenType::TokenDiv) {
        auto tokenNode = std::make_shared<TokenNode>();
        tokenNode->numIndex = -int(token.tokenType);

        ++index;
        CHECK(index < tokens_.size() && tokens_.at(index).tokenType == TokenType::TokenLeftBracket);

        ++index;
        CHECK(index < tokens_.size());
        const auto leftToken = tokens_.at(index);
        if (leftToken.tokenType == TokenType::TokenMul ||
            leftToken.tokenType == TokenType::TokenAdd ||
            leftToken.tokenType == TokenType::TokenNumber ||
            leftToken.tokenType == TokenType::TokenDiv) {
            tokenNode->left = generate_(index);
        } else {
            LOG(FATAL) << "unknown token type" << int(token.tokenType);
        }

        ++index;
        CHECK(index < tokens_.size() && tokens_.at(index).tokenType == TokenType::TokenComma);

        ++index;
        CHECK(index < tokens_.size());
        const auto rightToken = tokens_.at(index);
        if (rightToken.tokenType == TokenType::TokenMul ||
            rightToken.tokenType == TokenType::TokenAdd ||
            rightToken.tokenType == TokenType::TokenNumber ||
            rightToken.tokenType == TokenType::TokenDiv) {
            tokenNode->right = generate_(index);
        } else {
            LOG(FATAL) << "unknown token type" << int(token.tokenType);
        }

        ++index;
        CHECK(index < tokens_.size() && tokens_.at(index).tokenType == TokenType::TokenRightBracket);

        return tokenNode;
    } else {
        LOG(FATAL) << "unknown token type" << int(token.tokenType);
        return nullptr;
    }
}


}