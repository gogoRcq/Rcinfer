#include "layer/abstract/rcLayerRegister.h"
#include "common.h"
#include "glog/logging.h"
#include "layer/abstract/rcLayer.h"
#include "runtime/StateCode.h"
#include <memory>
#include <string>

namespace rq {

template<class T>
std::shared_ptr<rcLayer<T>> rcLayerRegister<T>::CreateLayer(const std::shared_ptr<RuntimeOperator<T>>& op) {
    CHECK(op != nullptr);
    CreatorRegistry &creatorRegistry = GetCreatorRegistry();
    const std::string& opType = op->type;
    LOG_IF(FATAL, creatorRegistry.count(opType) <= 0) << "cant find optype" << opType;

    const Creator& creator = creatorRegistry.at(opType);
    LOG_IF(FATAL, creator == nullptr) << "creator is empty";
    
    std::shared_ptr<rcLayer<T>> layer;
    const auto& status = creator(op, layer);
    LOG_IF(FATAL, status != ParseParamAttrStatus::rParameterAttrParseSuccess) << "creator error: " << opType << " error code: " << int(status);
    return layer;
}

template<class T>
void rcLayerRegister<T>::RegisterCreator(const std::string& layerType, const Creator& creator) {
    CHECK(creator != nullptr);
    CreatorRegistry &creatorRegistry = GetCreatorRegistry();
    CHECK(creatorRegistry.count(layerType) == 0) << "this type is allready registered";
    creatorRegistry.insert({layerType, creator});
}

INSTALLCLASS(rcLayerRegister);
INSTALLCLASS(rcLayerRegisterWapper);

}