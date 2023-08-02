#include "runtime/RuntimeGraph.h"
#include "common.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "runtime/RuntimeAttr.h"
#include "runtime/RuntimeDataType.h"
#include "runtime/RuntimeOperand.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/RuntimeParam.h"
#include "runtime/StateCode.h"
#include "runtime/ir.h"
#include <_types/_uint32_t.h>
#include <algorithm>
#include <memory>
#include <string>
#include <sys/_types/_int32_t.h>
#include <utility>
#include <vector>

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
    // 构建图关系
    for (const std::shared_ptr<RuntimeOperator<T>>& op : this->operators_) {
        const std::vector<std::string>& outputNames = op->outputNames;
        for (const std::shared_ptr<RuntimeOperator<T>>& op_ : this->operators_) {
            if (op == op_) continue;
            if (std::find(outputNames.begin(), outputNames.end(), op_->name) != outputNames.end()) {
                op->outputOperators.insert({op_->name, op_});
            }
        }
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
                std::shared_ptr<RuntimeParamBool> runtimeParam = std::make_shared<RuntimeParamBool>();
                runtimeParam->value = param.b;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterInt): {
                std::shared_ptr<RuntimeParamInt> runtimeParam = std::make_shared<RuntimeParamInt>();
                runtimeParam->value = param.i;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterFloat): {
                std::shared_ptr<RuntimeParamFloat> runtimeParam = std::make_shared<RuntimeParamFloat>();
                runtimeParam->value = param.f;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterString): {
                std::shared_ptr<RuntimeParamString> runtimeParam = std::make_shared<RuntimeParamString>();
                runtimeParam->value = param.s;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterIntArray): {
                std::shared_ptr<RuntimeParamIntArray> runtimeParam = std::make_shared<RuntimeParamIntArray>();
                runtimeParam->value = param.ai;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterFloatArray): {
                std::shared_ptr<RuntimeParamFloatArray> runtimeParam = std::make_shared<RuntimeParamFloatArray>();
                runtimeParam->value = param.af;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            case int(RuntimeParamType::rParameterStringArray): {
                std::shared_ptr<RuntimeParamStringArray> runtimeParam = std::make_shared<RuntimeParamStringArray>();
                runtimeParam->value = param.as;
                runtimeOperator->params.insert({name, runtimeParam});
                break;
            }
            default: {
                LOG(FATAL) << "unkown param type" << param.type;
            }
        }
    }
}

template<class T>
void RuntimeGraph<T>::Build(const std::string &inputName, const std::string &outputName) {
    if (this->graphState == GraphState::NeedInit) {
        bool ret = this->init();
        if (!ret) LOG(FATAL) << "Graph Init error";
    }
    if (this->graphState == GraphState::Complete) {
        return;
    }
    CHECK(this->graphState == GraphState::NeedBuild) << "graph state error";
    CHECK(!this->operators_.empty()) << "graph build failed for empty operators";

    this->inputOperators.clear();
    this->outputOperators.clear();

    for (const std::shared_ptr<RuntimeOperator<T>>& op : this->operators_) {
        if (op->type == "pnnx.Input") {
            this->inputOperators.insert({op->name, op});
        } else if (op->type == "pnnx.Output") {
            this->outputOperators.insert({op->name, op});
        } else {
            // 构建 layer
        }
    }
    RuntimeGraphShape<T>::InitOperatorInputTensor(this->operators_);
    RuntimeGraphShape<T>::InitOperatorOutputTensor(this->graph->ops, this->operators_);
    this->graphState = GraphState::Complete;
    this->inputName = inputName;
    this->outputName = outputName;
}

template<class T>
void RuntimeGraphShape<T>::InitOperatorInputTensor(const std::vector<std::shared_ptr<RuntimeOperator<T>>> &operators) {
    CHECK(!operators.empty()) << "error operators";
    for (const std::shared_ptr<RuntimeOperator<T>>& op : operators) {
        if (op->inputOperands.empty()) continue;
        const std::unordered_map<std::string, std::shared_ptr<RuntimeOperand<T>>>& inputOperands = op->inputOperands;
        for (const auto& inputOperands_iter : inputOperands) {
            const std::shared_ptr<RuntimeOperand<T>>& inputOperand = inputOperands_iter.second;
            const RuntimeDataType type = inputOperand->dataType;
            const std::vector<int32_t>& shapes = inputOperand->shapes;
            std::vector<std::shared_ptr<Tensor<T>>>& datas = inputOperand->datas;

            CHECK(!shapes.empty());
            const int32_t batchSize = shapes.at(0);
            CHECK(batchSize >= 0);
            CHECK(shapes.size() == 2 || shapes.size() == 3 || shapes.size() == 4) << "error batch size";
            if (!datas.empty()) {
                CHECK(batchSize == datas.size()) << "error batch size";
                for (int i = 0; i < batchSize; ++i) {
                    const std::vector<uint32_t>& data_shapes = datas.at(i)->shapes();
                    CHECK(data_shapes.size() == 3) << "the data shape not equal 3";
                    if (shapes.size() == 4) {
                        CHECK(data_shapes.at(0) == shapes.at(2));
                        CHECK(data_shapes.at(1) == shapes.at(3));
                        CHECK(data_shapes.at(2) == shapes.at(1));
                    } else if (shapes.size() == 3) {
                        CHECK(data_shapes.at(0) == shapes.at(1));
                        CHECK(data_shapes.at(1) == shapes.at(2));
                        CHECK(data_shapes.at(2) == 1);
                    } else {
                        CHECK(data_shapes.at(0) == shapes.at(1));
                        CHECK(data_shapes.at(1) == 1);
                        CHECK(data_shapes.at(2) == 1);
                    }
                }
            } else {
                datas.resize(batchSize);
                for (int i = 0; i < batchSize; ++i) {
                    if (shapes.size() == 4) {
                        datas.at(i) = std::make_shared<Tensor<T>>(shapes.at(2), shapes.at(3), shapes.at(1));
                    } else if (shapes.size() == 3) {
                        datas.at(i) = std::make_shared<Tensor<T>>(shapes.at(1), shapes.at(2), 1);
                    } else {
                        datas.at(i) = std::make_shared<Tensor<T>>(shapes.at(1), 1, 1);
                    }
                }
            }
        }
    }
}

template<class T>
void RuntimeGraphShape<T>::InitOperatorOutputTensor(const std::vector<pnnx::Operator *> &pnnx_operators, 
                                                    const std::vector<std::shared_ptr<RuntimeOperator<T>>> &operators) {
    CHECK(!pnnx_operators.empty() && !operators.empty());
    CHECK(pnnx_operators.size() == operators.size());
    for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
        const std::vector<pnnx::Operand *>& operands = pnnx_operators.at(i)->outputs;
        if (operands.empty()) continue;
        CHECK(operands.size() == 1) << "only one output is supported in rqinfer";

        pnnx::Operand *operand = operands.front();
        const std::shared_ptr<RuntimeOperator<T>> &runtime_op = operators.at(i);
        CHECK(operand != nullptr) << "Operand output is null";
        const std::vector<int32_t> &operand_shapes = operand->shape;
        const std::shared_ptr<RuntimeOperand<T>> &output_tensor = runtime_op->outputOperand;

        const int32_t batchSize = operand_shapes.at(0);
        CHECK(batchSize >= 0);
        CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 3 || operand_shapes.size() == 4) << "error batch size";
        
        if (!output_tensor) {
            std::shared_ptr<RuntimeOperand<T>> output_operand = std::make_shared<RuntimeOperand<T>>();
            output_operand->shapes = operand_shapes;
            output_operand->name = operand->name + "_output";
            switch (operand->type) {
                case 1: {
                    output_operand->dataType = RuntimeDataType::rTypeFloat32;
                    break;
                }
                case 0: {
                    output_operand->dataType = RuntimeDataType::rTypeUnknown;
                    break;
                }
                default: {
                    LOG(FATAL) << "this type is not supported now: " << operand->type;
                }
            }
            for (int j = 0; j < batchSize; ++j) {
                if (operand_shapes.size() == 4) {
                    output_operand->datas.emplace_back(std::make_shared<Tensor<T>>(operand_shapes.at(2), operand_shapes.at(3), operand_shapes.at(1)));
                } else if (operand_shapes.size() == 3) {
                    output_operand->datas.emplace_back(std::make_shared<Tensor<T>>(operand_shapes.at(1), operand_shapes.at(2), 1));
                } else {
                    output_operand->datas.emplace_back(std::make_shared<Tensor<T>>(operand_shapes.at(1), 1, 1));
                }
            }
            runtime_op->outputOperand = std::move(output_operand);
        } else {
            CHECK(output_tensor->datas.size() == batchSize);
            CHECK(output_tensor->shapes == operand_shapes); // operand_shapes : batch channel row col
            for (int j = 0; j < batchSize; ++j) {
                const std::vector<uint32_t> &tensor_shapes = output_tensor->datas.at(j)->shapes(); // row col channel
                if (operand_shapes.size() == 4) {
                    if (tensor_shapes.at(0) != operand_shapes.at(2) ||
                        tensor_shapes.at(1) != operand_shapes.at(3) ||
                        tensor_shapes.at(2) != operand_shapes.at(1)) {
                        DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
                        const std::vector<uint32_t> target_shapes = {(uint32_t)operand_shapes.at(2),
                                                                     (uint32_t)operand_shapes.at(3),
                                                                     (uint32_t)operand_shapes.at(1)};
                        output_tensor->datas.at(j)->ReRawshape(target_shapes);
                    }
                } else if (operand_shapes.size() == 3) {
                    if (tensor_shapes.at(0) != operand_shapes.at(1) ||
                        tensor_shapes.at(1) != operand_shapes.at(2) ||
                        tensor_shapes.at(2) != 1) {
                        DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
                        const std::vector<uint32_t> target_shapes = {(uint32_t)operand_shapes.at(1),
                                                                     (uint32_t)operand_shapes.at(2),
                                                                     1};
                        output_tensor->datas.at(j)->ReRawshape(target_shapes);
                    }
                } else {
                    if (tensor_shapes.at(0) != operand_shapes.at(1) ||
                        tensor_shapes.at(1) != 1 ||
                        tensor_shapes.at(2) != 1) {
                        DLOG(WARNING) << "The shape of tensor do not adapting with output operand";
                        const std::vector<uint32_t> target_shapes = {(uint32_t)operand_shapes.at(1),
                                                                     1,
                                                                     1};
                        output_tensor->datas.at(j)->ReRawshape(target_shapes);
                    }
                }
            }
        }
    }
}

template<class T>
bool RuntimeGraph<T>::CheckOperatorReady(const std::shared_ptr<RuntimeOperator<T>> &op) {
    CHECK(op != nullptr);
    CHECK(op->meetNum <= op->inputOperands.size());
    return op->meetNum == op->inputOperands.size();
}

template<class T>
void RuntimeGraph<T>::ProbeNextLayer(const std::shared_ptr<RuntimeOperator<T>> &currentOp, 
                                     std::deque<std::shared_ptr<RuntimeOperator<T>>> &operatorQueue,
                                     std::vector<std::shared_ptr<Tensor<T>>> layerOutputData) {
    std::unordered_map<std::string, std::shared_ptr<RuntimeOperator<T>>>& outputOperators = currentOp->outputOperators;
    std::vector<std::vector<std::shared_ptr<Tensor<T>>>> nextLayerDatas;
    for (const auto& outputOperator : outputOperators) {
        auto& nextOp = outputOperator.second;
        std::unordered_map<std::string, std::shared_ptr<RuntimeOperand<T>>>& nextInputOperands = nextOp->inputOperands;
        if (nextInputOperands.find(currentOp->name) != nextInputOperands.end()) {
            std::vector<std::shared_ptr<Tensor<T>>>& nextInputOperandTensors = nextInputOperands[currentOp->name]->datas;
            nextLayerDatas.push_back(nextInputOperandTensors);
            nextOp->meetNum++;
            if (std::find(operatorQueue.begin(), operatorQueue.end(), nextOp) == operatorQueue.end()) {
                if (CheckOperatorReady(nextOp)) {
                    operatorQueue.push_back(nextOp);
                }
            }
        }
    }
    SetOpInputData(layerOutputData, nextLayerDatas);
}

template<class T>
void RuntimeGraph<T>::SetOpInputData(std::vector<std::shared_ptr<Tensor<T>>> &src,
                                     std::vector<std::vector<std::shared_ptr<Tensor<T>>>> &dst) {
    CHECK(!src.empty() && !dst.empty());
    for (int i = 0; i < src.size(); ++i) {
        for (int j = 0; j < dst.size(); ++j) {
            dst[j][i]->setdata(src[i]->data());
        }
    }                           
}

template<class T>
std::vector<std::shared_ptr<Tensor<T>>> RuntimeGraph<T>::forward(const std::vector<std::shared_ptr<Tensor<T>>> &inputs, bool debug) {
    if (this->graphState < GraphState::Complete) {
        LOG(FATAL) << "graph need build";
    }
    CHECK(this->graphState == GraphState::Complete) << "graph state error";
    
    std::shared_ptr<RuntimeOperator<T>> inputOp;
    if (this->inputOperators.find(this->inputName) != this->inputOperators.end()) {
        inputOp = this->inputOperators[this->inputName];
    } else {
        LOG(FATAL) << "cant find inputOp";
    }

    std::shared_ptr<RuntimeOperator<T>> outputOp;
    if (this->outputOperators.find(this->outputName) != this->outputOperators.end()) {
        outputOp = this->outputOperators[this->outputName];
    } else {
        LOG(FATAL) << "cant find outputOp";
    }

    std::deque<std::shared_ptr<RuntimeOperator<T>>> operatorQueue;
    operatorQueue.push_back(inputOp);
    std::map<std::string, double> run_duration_infos;// what ?

    while (!operatorQueue.empty()) {
        std::shared_ptr<RuntimeOperator<T>> currentOp = operatorQueue.front();
        operatorQueue.pop_front();

        if (!currentOp || currentOp == outputOp) {
            LOG(INFO) << "Model Inference End";
            break;
        }

        if (currentOp == inputOp) {
            ProbeNextLayer(currentOp, operatorQueue, inputs);
        } else {
            std::string curOpName = currentOp->name;
            if (!CheckOperatorReady(currentOp)) {
                if (operatorQueue.empty()) {
                    LOG(FATAL) << "cant ready"; // 如果已经在最后一个节点，则无法 ready
                } else {
                    operatorQueue.push_back(currentOp); // 如果不是最后一个节点还有可能
                    continue;
                }
            }
            std::vector<std::shared_ptr<RuntimeOperand<T>>>& curInputOperandsSeq = currentOp->inputOperandsSeq;
            std::vector<std::shared_ptr<Tensor<T>>> layerInputDatas;
            for (const std::shared_ptr<RuntimeOperand<T>>& curInputOperand : curInputOperandsSeq) {
                for (const std::shared_ptr<Tensor<T>>& curInputTensor : curInputOperand->datas) {
                    layerInputDatas.push_back(curInputTensor);
                }
            }

            CHECK(!layerInputDatas.empty()) << "layer input data is empty";
            CHECK(currentOp->outputOperand != nullptr && !currentOp->outputOperand->datas.empty()) << "layer output data is empty";
            const auto &start = std::chrono::steady_clock::now();
            // excute

            ProbeNextLayer(currentOp, operatorQueue, currentOp->outputOperand->datas);
            if (debug) {
                LOG(INFO) << "current operator: " << currentOp->name;
            }
        }
    }

    for (std::shared_ptr<RuntimeOperator<T>>& op : this->operators_) {
        op->meetNum = 0;
    }

    CHECK(outputOp->inputOperands.size() == 1) << "The graph only support one path to the output node yet!";
    const auto& outOperandIter = outputOp->inputOperands.begin();
    const auto& outOperand = outOperandIter->second;
    return outOperand->datas;
}

INSTALLCLASS(RuntimeGraph);
INSTALLCLASS(RuntimeGraphShape);
}