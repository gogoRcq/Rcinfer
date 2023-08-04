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
    rcLinearLayer(int32_t inFeatures, int32_t outFeatures, bool useBias) : 
                  inFeatures(inFeatures), outFeatures(outFeatures), useBias(useBias) {
        if (useBias) this->initBias(1, outFeatures, 1, 1);
    }

private:
    int32_t inFeatures = 0;
    int32_t outFeatures = 0;
    bool useBias = false;
};

}

#endif