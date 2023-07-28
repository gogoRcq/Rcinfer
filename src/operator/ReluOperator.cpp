#include "operator/ReluOperator.h"
#include "common.h"

namespace rq {

template<class T>
void ReluOperator<T>::setThresh(T val) {
    thresh = val;
}

template<class T>
T ReluOperator<T>::getThresh() const {
    return thresh;
}

INSTALLCLASS(ReluOperator);

}