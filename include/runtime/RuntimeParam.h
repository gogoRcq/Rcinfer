#ifndef RUNTIMEPARAM_H_
#define RUNTIMEPARAM_H_

#include <vector>
#include <iostream>
#include <string>
#include "StateCode.h"

namespace rq{

class RuntimeParam {
public:
    RuntimeParamType type;
    virtual ~RuntimeParam() = default;
    explicit RuntimeParam(RuntimeParamType type = RuntimeParamType::rParameterUnknown) : type(type){};
};

class RuntimeParamBool : public RuntimeParam {
public:
    virtual ~RuntimeParamBool() = default;
    RuntimeParamBool() : RuntimeParam(RuntimeParamType::rParameterBool){};
    bool value = false;
};

class RuntimeParamInt : public RuntimeParam {
public:
    virtual ~RuntimeParamInt() = default;
    RuntimeParamInt() : RuntimeParam(RuntimeParamType::rParameterInt){};
    int value = 0;
};

class RuntimeParamFloat : public RuntimeParam {
public:
    virtual ~RuntimeParamFloat() = default;
    RuntimeParamFloat() : RuntimeParam(RuntimeParamType::rParameterFloat){};
    float value = 0.;
};

class RuntimeParamString : public RuntimeParam {
public:
    virtual ~RuntimeParamString() = default;
    RuntimeParamString() : RuntimeParam(RuntimeParamType::rParameterString){};
    std::string value;
};

class RuntimeParamIntArray : public RuntimeParam {
public:
    virtual ~RuntimeParamIntArray() = default;
    RuntimeParamIntArray() : RuntimeParam(RuntimeParamType::rParameterIntArray){};
    std::vector<int> value;
};

class RuntimeParamFloatArray  : public RuntimeParam {
public:
    virtual ~RuntimeParamFloatArray () = default;
    RuntimeParamFloatArray () : RuntimeParam(RuntimeParamType::rParameterFloatArray ){};
    std::vector<float> value;
};

class RuntimeParamStringArray  : public RuntimeParam {
public:
    virtual ~RuntimeParamStringArray () = default;
    RuntimeParamStringArray () : RuntimeParam(RuntimeParamType::rParameterStringArray ){};
    std::vector<std::string> value;
};

}

#endif