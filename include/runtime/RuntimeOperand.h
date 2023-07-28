#ifndef RUNTIMEOPERAND_H_
#define RUNTIMEOPERAND_H_

#include <iostream>
#include <string>
#include <vector>
#include "RuntimeDataType.h"
#include "data/Tensor.h"

namespace rq{

// 操作数的抽象
template<class T>
class RuntimeOperand {
public:
    std::string name; // 名称
    std::vector<int32_t> shapes; // 数据 shape
    std::vector<std::shared_ptr<Tensor<T>>> datas; // 数据
    RuntimeDataType dataType = RuntimeDataType::rTypeUnknown; // 数据类型
};

}


#endif