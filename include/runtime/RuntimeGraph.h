#ifndef RUNTIMEGRAPH_H_
#define RUNTIMEGRAPH_H_

#include "RuntimeOperator.h"
#include "ir.h"
#include <deque>

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
    void Build(const std::string &inputName, const std::string &outputName);
    const std::vector<std::shared_ptr<RuntimeOperator<T>>> operators() const;
    std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>> &inputs, bool debug = false);


private:
    static void InitInputOperators(const std::vector<pnnx::Operand*> &inputs,
                                   const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void InitOutputOperators(const std::vector<pnnx::Operand*> &outputs,
                                    const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                               const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                const std::shared_ptr<RuntimeOperator<T>> &runtimeOperator);
    static void ProbeNextLayer(const std::shared_ptr<RuntimeOperator<T>> &current_op, 
                               std::deque<std::shared_ptr<RuntimeOperator<T>>> &operator_queue,
                               std::vector<std::shared_ptr<Tensor<T>>> layer_output_data);
    static bool CheckOperatorReady(const std::shared_ptr<RuntimeOperator<T>> &op);
    static void SetOpInputData(std::vector<std::shared_ptr<Tensor<T>>> &src,
                               std::vector<std::vector<std::shared_ptr<Tensor<T>>>> &dst);
};

template<class T>
class RuntimeGraphShape {
 public:
  /**
   * 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输入operand的形状和operand中张量的形状是否匹配
   * @param operators 计算图中的计算节点
   */
  static void InitOperatorInputTensor(const std::vector<std::shared_ptr<RuntimeOperator<T>>> &operators);

  /**
   * 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输出operand的形状和operand中张量的形状是否匹配
   * @param pnnx_operators pnnx图节点
   * @param operators rqInfer计算图中的计算节点
   */
  static void InitOperatorOutputTensor(const std::vector<pnnx::Operator *> &pnnx_operators,
                                       const std::vector<std::shared_ptr<RuntimeOperator<T>>> &operators);
};

}

#endif