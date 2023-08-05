#ifndef RCLAYERREGISTER_H_
#define RCLAYERREGISTER_H_


#include "runtime/RuntimeOperator.h"
#include "rcLayer.h"
#include "runtime/StateCode.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace rq {

template<class T>
class rcLayerRegister {
public:
    typedef ParseParamAttrStatus (*Creator) (const std::shared_ptr<RuntimeOperator<T>>& op, 
                                             std::shared_ptr<rcLayer<T>>& layer);
    
    typedef std::unordered_map<std::string, Creator> CreatorRegistry; // 名称对应创建函数

    static void RegisterCreator(const std::string& layerType, const Creator& creator);

    static CreatorRegistry& GetCreatorRegistry() {
        static CreatorRegistry creatorRegistry;
        return creatorRegistry;
    }

    static std::shared_ptr<rcLayer<T>> CreateLayer(const std::shared_ptr<RuntimeOperator<T>>& op);
};

template<class T>
class rcLayerRegisterWapper {
public:
    rcLayerRegisterWapper(const std::string& layerType, const typename rcLayerRegister<T>::Creator& creator) {
        rcLayerRegister<T>::RegisterCreator(layerType, creator);
    }
};

#define RCREGISTER_CREATOR(name, layertype, layer)\
    static rcLayerRegisterWapper<float> g_creator_f_##name(layertype, layer<float>::creatorInstance)

}


#endif