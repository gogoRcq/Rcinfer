#ifndef PRASEEXPRESSION_H_
#define PRASEEXPRESSION_H_

#include <iostream>
#include <memory>
#include <string>
#include <sys/_types/_int32_t.h>

namespace rq {

enum class TokenType {
    TokenUnknown = -1,
    TokenNumber = 0,
    TokenComma = 1,
    TokenAdd = 2,
    TokenMul = 3,
    TokenLeftBracket = 4,
    TokenRightBracket = 5,
    TokenDiv = 6
};

class Token {
public:
    TokenType tokenType = TokenType::TokenUnknown;
    int32_t startPos = 0;
    int32_t endPos = 0;
    Token(TokenType tokenType, int32_t startPos, int32_t endPos) : tokenType(tokenType), startPos(startPos), endPos(endPos) {};
};

class TokenNode {
public:
    std::shared_ptr<TokenNode> left = nullptr;
    std::shared_ptr<TokenNode> right = nullptr;
    int32_t numIndex = -1; // 负数表示非操作数编号，是 tokentype 的相反数，非负数就是数据的编号
    TokenNode(int32_t numIndex, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right) : numIndex(numIndex), left(left), right(right) {}
    TokenNode(){}
};

class ExpressionParser {
public:
    explicit ExpressionParser(std::string statement) : statement(std::move(statement)){}
    std::vector<std::shared_ptr<TokenNode>> generate();
    void tokenizer(bool needRetoken = false);
    const std::vector<Token>& tokens() const;
    const std::vector<std::string>& tokenStrings() const;
private:
    std::string statement;
    std::vector<Token> tokens_;
    std::vector<std::string> tokenStrings_;
    std::shared_ptr<TokenNode> generate_(int32_t& index);
    void ReversePolish(std::shared_ptr<TokenNode>& curNode, std::vector<std::shared_ptr<TokenNode>>& reversePolish);
};

}

#endif