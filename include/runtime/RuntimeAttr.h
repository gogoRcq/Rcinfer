#ifndef RUNTIMEATTR_H_
#define RUNTIMEATTR_H_

#include <vector>
#include "RuntimeDataType.h"
#include "glog/logging.h"

namespace rq {

template<class T>
class RuntimeAttr {
public:
    std::vector<char> weightData;
    std::vector<int32_t> shape;
    RuntimeDataType dataType = RuntimeDataType::rTypeUnknown;
    std::vector<T> get(); // 获取权重参数
};

template<class T>
std::vector<T> RuntimeAttr<T>::get() {
    CHECK(!weightData.empty());
    CHECK(dataType != RuntimeDataType::rTypeUnknown);
    std::vector<T> weights;
    switch (this->type){
        case RuntimeDataType::rTypeFloat32: {
            const bool isFloat = std::is_same<T, float>::value;
            CHECK_EQ(isFloat, true);
            const int floatSize = sizeof(float);
            CHECK_EQ(weightData.size() % floatSize, 0);
            const uint32_t weightNum = weightData.size() / floatSize;
            weights.reserve(weightNum);
            for (uint32_t i = 0; i < weights; ++i) {
                weights.emplace_back(*(((float *)weightData.data()) + i));
            }
        }
    default:
        LOG(FATAL) << "this type is not supported now";
    }
}

}

#endif