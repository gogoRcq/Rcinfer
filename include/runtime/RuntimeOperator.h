#ifndef RUNTIMEOPERATOR_H_
#define RUNTIMEOPERATOR_H_

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "RuntimeAttr.h"
#include "RuntimeOperand.h"
#include "RuntimeParam.h"
#include "LayerRegister.h"

namespace rq {

template <class T>
class RuntimeOperator {
public:
    int meetNum = 0;
    std::string name; /// 计算节点的名称
    std::string type; /// 计算节点的类型
    std::shared_ptr<Layer<T>> layer; // 该节点对应的计算层

    std::vector<std::string> outputNames; // 输出节点名
    std::shared_ptr<RuntimeOperand<T>> outputOperand; // 该节点的输出操作数，算出来只有一个，所以不需要vector
    std::unordered_map<std::string, std::shared_ptr<RuntimeOperator<T>>> outputOperators; // 输出节点名和节点对应

    std::vector<std::shared_ptr<RuntimeOperand<T>>> inputOperandsSeq; // 节点的输入操作数 顺序存放
    std::unordered_map<std::string, std::shared_ptr<RuntimeOperand<T>>>  inputOperands; // 节点的输入操作数 

    std::unordered_map<std::string, std::shared_ptr<RuntimeParam>> params; // 算子的参数信息
    std::unordered_map<std::string, std::shared_ptr<RuntimeAttr<T>>> attributes; // 算子的权重信息
};

}

#endif