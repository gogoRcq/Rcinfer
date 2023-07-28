#ifndef LAYERREGISTER_H_
#define LAYERREGISTER_H_

#include <iostream>
#include <map>
#include "operator/Operator.h"
#include "layer/Layer.h"

namespace rq {

template<class T>
class LayerRegister {
public:
    typedef std::shared_ptr<Layer<T>> (*Creator) (const std::shared_ptr<Operator>& op);

    typedef std::map<OperatorType, Creator> layerRegistry;

    static layerRegistry& getLayerRegistry() {
        static layerRegistry registry;
        return registry;
    }

    static void registerOperator(OperatorType opType, const Creator& creator);

    static std::shared_ptr<Layer<T>> creatorLayer(const std::shared_ptr<Operator>& op);
};


template<class T>
class LayerRegisterWapper {
public:
    LayerRegisterWapper(OperatorType opType, const typename LayerRegister<T>::Creator& creator) {
        LayerRegister<T>::registerOperator(opType, creator);
    }
};


#define REGISTER_CREATOR(name, optype, layer)\
    static LayerRegisterWapper<float> g_creator_f_##name(optype, layer<float>::creatorInstance)

}


#endif