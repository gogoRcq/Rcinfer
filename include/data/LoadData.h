#ifndef LOADDATA_H_
#define LOADDATA_H_

#include <iostream>
#include <fstream>
#include <armadillo>
#include <fstream>
#include "glog/logging.h"
#include "Tensor.h"

namespace rq {

template<class T>
class CSVDataLoader {
public:
    static std::shared_ptr<Tensor<T>> loadData(const std::string &filePath, char splitChar = ',');

    static std::shared_ptr<Tensor<T>> loadDataWithHeader(const std::string &filePath, std::vector<std::string> &headers, 
                                                         char splitChar = ',');
private:
    static std::pair<size_t, size_t> getMatrixSize(std::ifstream &file, char splitChar);
};

}

#endif