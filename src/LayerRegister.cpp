#include "LayerRegister.h"
#include "common.h"
#include "layer/ReluLayer.h"

namespace rq {

template<class T>
void LayerRegister<T>::registerOperator(OperatorType opType, const Creator& creator) {
    CHECK(creator != nullptr) << "error operator creator!";
    layerRegistry& registry = getLayerRegistry();
    CHECK_EQ(registry.count(opType), 0) << "operator allready regiter!";
    registry.insert({opType, creator});
}

template<class T>
std::shared_ptr<Layer<T>> LayerRegister<T>::creatorLayer(const std::shared_ptr<Operator>& op) {
    layerRegistry& registry = getLayerRegistry();
    const OperatorType opType = op->opType;
    LOG_IF(FATAL, registry.count(opType) <= 0) << "no such operator!";
    
    const Creator& creator = registry.at(opType);
    LOG_IF(FATAL, creator == nullptr) << "error creator!";

    std::shared_ptr<Layer<T>> layer = creator(op);
    LOG_IF(FATAL, !layer) << "init layer error!";
    return layer;
}

INSTALLCLASS(LayerRegister);
INSTALLCLASS(LayerRegisterWapper);
// register


}