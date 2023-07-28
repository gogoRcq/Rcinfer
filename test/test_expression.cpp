//
// Created by fss on 23-1-15.
//

#include "runtime/PraseExpression.h"
#include <gtest/gtest.h>
#include <glog/logging.h>


// static void ShowNodes(const std::shared_ptr<rq::TokenNode> &node) {
//   if (!node) {
//     return;
//   }
//   // 中序遍历的顺序
//   ShowNodes(node->left);
//   if (node->numIndex < 0) {
//     if (node->numIndex == -int(rq::TokenType::TokenAdd)) {
//       LOG(INFO) << "ADD";
//     } else if (node->numIndex == -int(rq::TokenType::TokenMul)) {
//       LOG(INFO) << "MUL";
//     } else if (node->numIndex == -int(rq::TokenType::TokenDiv)) {
//       LOG(INFO) << "DIV";
//     }
//   } else {
//     LOG(INFO) << "NUM: " << node->numIndex;
//   }
//   ShowNodes(node->right);
// }

// TEST(test_expression, expression1) {
//   using namespace rq;
//   const std::string &statement = "add(@1,@2)";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.generate();
//   ShowNodes(node_tokens);
// }

// TEST(test_expression, expression3) {
//   using namespace rq;
//   const std::string &statement = "add(mul(@0,@1),mul(@2,add(@3,@4)))";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.generate();
//   ShowNodes(node_tokens);
// }

// TEST(test_expression, expression4) {
//   using namespace rq;
//   const std::string &statement = "add(mul(@0,@1),mul(@2,@3))";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.generate();
//   ShowNodes(node_tokens);
// }

// TEST(test_expression, expression5) {
//   using namespace rq;
//   const std::string &statement = "add(div(@0,@1),@2)";
//   ExpressionParser parser(statement);
//   const auto &node_tokens = parser.generate();
//   ShowNodes(node_tokens);
// }

