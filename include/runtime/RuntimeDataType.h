#ifndef RUNTIMEDATATYPE_H_
#define RUNTIMEDATATYPE_H_

namespace rq{

enum class RuntimeDataType {
    rTypeUnknown = 0,
    rTypeFloat32 = 1,
    rTypeFloat64 = 2,
    rTypeFloat16 = 3,
    rTypeInt32 = 4,
    rTypeInt64 = 5,
    rTypeInt16 = 6,
    rTypeInt8 = 7,
    rTypeUInt8 = 8,
};

}

#endif