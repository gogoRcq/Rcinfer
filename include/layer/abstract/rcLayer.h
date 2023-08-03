#ifndef RCLAYER_H_
#define RCLAYER_H_


#include "data/Tensor.h"
#include <memory>
#include <string>
#include <vector>
#include "runtime/StateCode.h"

namespace rq {

template<class T>
class rcLayer {
public:
    explicit rcLayer(const std::string& layerName) : layerName(layerName) {};

    virtual ~rcLayer() = default;

    virtual const std::string& getLayerName() {return this->layerName;};

    virtual void setBias(const std::vector<std::shared_ptr<Tensor<T>>>& bias);

    virtual void setBias(const std::vector<T>& bias);

    virtual void setWights(const std::vector<std::shared_ptr<Tensor<T>>>& weights);

    virtual void setWights(const std::vector<T>& weights);

    virtual const std::vector<std::shared_ptr<Tensor<T>>>& getWights() const;

    virtual const std::vector<std::shared_ptr<Tensor<T>>>& getBias() const;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs);

private:
    std::string layerName;
};

}

#endif