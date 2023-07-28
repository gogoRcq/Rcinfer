#include "runtime/RuntimeGraph.h"
#include "common.h"
#include "runtime/RuntimeAttr.h"
#include "runtime/RuntimeDataType.h"
#include "runtime/RuntimeOperand.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/RuntimeParam.h"
#include "runtime/StateCode.h"
#include "runtime/ir.h"
#include <memory>
#include <string>
#include <utility>

namespace rq {

template<class T>
void RuntimeGraph<T>::setParamPath(const std::string& paramPath) {
    this->paramPath = paramPath;
}

template<class T>
void RuntimeGraph<T>::setBinPath(const std::string& binPath) {
    this->binPath = binPath;
}

template<class T>
const std::string& RuntimeGraph<T>::getParamPath() {
    return this->paramPath;
}

template<class T>
const std::string& RuntimeGraph<T>::setBinPath() {
    return this->binPath;
}

template<class T>
const std::vector<std::shared_ptr<RuntimeOperator<T>>> RuntimeGraph<T>::operators() const {
    return this->operators_;
}

template<class T>
bool RuntimeGraph<T>::init() {
    if (this->binPath.empty() || this->paramPath.empty()) {
        LOG(ERROR) << "empty binpath or parampath";
        return false;
    }

    this->graph = std::make_unique<pnnx::Graph>();
    int ret = this->graph->load(this->paramPath, this->binPath);
    if (ret != 0) {
        LOG(ERROR) << "error binpath or parampath: " << this->paramPath << " " << this->binPath;
        return false;
    }
    std::vector<pnnx::Operator*> operators = this->graph->ops;
    if (operators.empty()) {
        LOG(ERROR) << "Can not read the layers' define";
        return false;
    }
    this->operators_.clear();
    for (const pnnx::Operator* op : operators) {
        if (!op) {
            LOG(ERROR) << "meet empty node";
            continue;
        }

        std::shared_ptr<RuntimeOperator<T>> runtimeOperator = std::make_shared<RuntimeOperator<T>>();
        runtimeOperator->name = op->name;
        runtimeOperator->type = op->type;

        const auto& outputs = op->outputs;
        if (!outputs.empty()) {
            InitOutputOperators(outputs, runtimeOperator);
        }

        const auto& inputs = op->inputs;
        if (!inputs.empty()) {
            InitInputOperators(inputs, runtimeOperator);
        } 

        const auto& attrs = op->attrs;
        if (!attrs.empty()) {
            InitGraphAttrs(attrs, runtimeOperator);
        }

        const auto& params = op->params;
        if (!params.empty()) {
            InitGraphParams(params, runtimeOperator);
        }

        this->operators_.emplace_back(runtimeOperator);
    }
    graphState = GraphState::NeedBuild;
    return true;
}

template<class T>
void RuntimeGraph<T>::InitInputOperators(const std::vector<pnnx::Operand*> &inputs,
                                         const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator) {
    for (const pnnx::Operand* input : inputs) {
        if (!input) continue;
        const pnnx::Operator* producer = input->producer;
        std::shared_ptr<RuntimeOperand<T>> rInput = std::make_shared<RuntimeOperand<T>>();
        rInput->name = producer->name;
        rInput->shapes = input->shape;
        switch (input->type) {
            case 1: {
                rInput->dataType = RuntimeDataType::rTypeFloat32;
                break;
            }
            case 0: {
                rInput->dataType = RuntimeDataType::rTypeUnknown;
                break;
            }
            default: {
                LOG(FATAL) << "this type is not supported now: " << input->type;
            }
        }
        runtimeOperator->inputOperands.insert({rInput->name, rInput});
        runtimeOperator->inputOperandsSeq.emplace_back(rInput);
    }
}

template<class T>
void RuntimeGraph<T>::InitOutputOperators(const std::vector<pnnx::Operand*> &outputs,
                                          const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator) {
    // 为什么只需要初始化输出操作数的名称呢？
    for (const pnnx::Operand* output : outputs) {
        if (!output) continue;
        const std::vector<pnnx::Operator*>& consumers = output->consumers;
        for (const pnnx::Operator* consumer : consumers) {
            runtimeOperator->outputNames.emplace_back(consumer->name);
        }
    }
}

template<class T>
void RuntimeGraph<T>::InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                     const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator) {
    for (auto& pair : attrs) {
        const std::string& name = pair.first;
        const pnnx::Attribute& attr = pair.second;
        switch (attr.type) {
            case 1 : {
                std::shared_ptr<RuntimeAttr<T>> runtimeAttr = std::make_shared<RuntimeAttr<T>>();
                runtimeAttr->weightData = attr.data;
                runtimeAttr->shape = attr.shape;
                runtimeAttr->dataType = RuntimeDataType::rTypeFloat32;
                runtimeOperator->attributes.insert({name, runtimeAttr});
                break;
            }
            default: {
                LOG(FATAL) << "this type is not supported now: " << attr.type;
            }
        }
    }
}

template<class T>
void RuntimeGraph<T>::InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                      const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator) {
    for (auto& pair : params) {
        const std::string& name = pair.first;
        const pnnx::Parameter& param = pair.second;
        switch (param.type) {
            case int(RuntimeParamType::rParameterUnknown): {
                    std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParam>();
                    runtimeOperator->params.insert({name, runtimeParam});
                    break;
            }
            case int(RuntimeParamType::rParameterBool): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamBool>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterInt): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamInt>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterFloat): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamFloat>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterString): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamString>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterIntArray): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamIntArray>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterFloatArray): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamFloatArray>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterStringArray): {
                std::shared_ptr<RuntimeParam> runtimeParam = std::make_shared<RuntimeParamStringArray>();
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            default: {
                LOG(FATAL) << "unkown param type" << param.type;
            }
        }
    }
}
INSTALLCLASS(RuntimeGraph);

}