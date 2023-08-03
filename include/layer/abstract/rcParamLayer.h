#ifndef RCPARAMLAYER_H_
#define RCPARAMLAYER_H_

#include "glog/logging.h"
#include "rcLayer.h"
#include <_types/_uint32_t.h>
#include <string>

namespace rq {

template<class T>
class rcParamLayer : public rcLayer<T> {
public:
    explicit rcParamLayer(const std::string& layerName) : rcLayer<T>(layerName) {};

    ~rcParamLayer() override = default;

    void setBias(const std::vector<std::shared_ptr<Tensor<T>>>& bias) override;

    void setBias(const std::vector<T>& bias) override;

    void setWights(const std::vector<std::shared_ptr<Tensor<T>>>& weights) override;

    void setWights(const std::vector<T>& weights) override;

    const std::vector<std::shared_ptr<Tensor<T>>>& getWights() const override;

    const std::vector<std::shared_ptr<Tensor<T>>>& getBias() const override;

    void initBias(uint32_t paramCount, uint32_t paramRow, uint32_t paramCol, uint32_t paramChannel);

    void initWights(uint32_t paramCount, uint32_t paramRow, uint32_t paramCol, uint32_t paramChannel);

private:
    std::vector<std::shared_ptr<Tensor<T>>> weights;
    std::vector<std::shared_ptr<Tensor<T>>> bias;
};

}

#endif