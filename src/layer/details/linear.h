#ifndef LINEAR_H_
#define LINEAR_H_

#include "layer/abstract/rcParamLayer.h"
#include "runtime/RuntimeOperator.h"
#include "runtime/StateCode.h"
#include <_types/_uint32_t.h>
#include <sys/_types/_int32_t.h>
#include <sys/types.h>

namespace rq {

template<class T>
class rcLinearLayer : public rcParamLayer<T> {
public:
    rcLinearLayer(int32_t inFeatures, int32_t outFeatures, bool useBias) : rcParamLayer<T>("linear"),
                  inFeatures(inFeatures), outFeatures(outFeatures), useBias(useBias) {
        if (useBias) this->initBias(1, outFeatures, 1, 1);
        this->initWights(1, outFeatures, inFeatures, 1);
    }

    ~rcLinearLayer() override = default;

    virtual InferStatus forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<T>>> &outputs) override;

    static ParseParamAttrStatus creatorInstance(const std::shared_ptr<RuntimeOperator<T>> &op,
                                                std::shared_ptr<rcLayer<T>>& layer);

private:
    int32_t inFeatures = 0;
    int32_t outFeatures = 0;
    bool useBias = false;
};

}

#endif