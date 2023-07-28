#ifndef RUNTIMEGRAPH_H_
#define RUNTIMEGRAPH_H_

#include "RuntimeOperator.h"
#include "ir.h"

namespace rq {

template<class T>
class RuntimeGraph {
private:
    enum class GraphState {
        NeedInit = -2,
        NeedBuild = -1,
        Complete = 0,
    };
    GraphState graphState = GraphState::NeedInit;
    std::string inputName; /// 计算图输入节点的名称
    std::string outputName; /// 计算图输出节点的名称
    std::string paramPath; /// 计算图的结构文件
    std::string binPath; /// 计算图的权重文件
    std::unordered_map<std::string, std::shared_ptr<RuntimeOperator<T>>> inputOperators; // 保存输入节点
    std::unordered_map<std::string, std::shared_ptr<RuntimeOperator<T>>> outputOperators; // 保存输出节点
    std::vector<std::shared_ptr<RuntimeOperator<T>>> operators_; // 计算图的计算节点
    std::unique_ptr<pnnx::Graph> graph; // pnnx生成的计算图

public:
    RuntimeGraph(std::string paramPath, std::string binPath) : paramPath(std::move(paramPath)), binPath(std::move(binPath)){};
    ~RuntimeGraph() = default;


    void setParamPath(const std::string& paramPath);
    void setBinPath(const std::string& binPath);
    const std::string& getParamPath();
    const std::string& setBinPath();
    bool init();
    const std::vector<std::shared_ptr<RuntimeOperator<T>>> operators() const;

private:
    static void InitInputOperators(const std::vector<pnnx::Operand*> &inputs,
                                   const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void InitOutputOperators(const std::vector<pnnx::Operand*> &outputs,
                                    const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                               const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
};

}

#endif