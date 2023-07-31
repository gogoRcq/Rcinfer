#ifndef TENSOR_H_
#define TENSOR_H_

#include <armadillo>
#include <iostream>
#include <vector>
#include "glog/logging.h"

namespace rq{

template<class T>
class Tensor {
private:
    std::vector<uint32_t> rawShapes;
    arma::Cube<T> tsData;
public:
    explicit Tensor() = default;
    Tensor(uint32_t row, uint32_t col, uint32_t channel);
    Tensor(const Tensor<T> &tensor);
    Tensor<T>& operator=(const Tensor<T> &tensor);
    ~Tensor() = default;

    uint32_t rows() const;

    uint32_t cols() const;

    uint32_t channels() const;

    uint64_t size() const;

    bool empty() const;

    std::vector<uint32_t> shapes() const;

    arma::Cube<T>& data();

    const arma::Cube<T>& data() const;

    void setdata(arma::Cube<T>& data);

    T at(uint32_t channel, uint32_t row, uint32_t col) const;

    T& at(uint32_t channel, uint32_t row, uint32_t col);

    T index(uint64_t offset) const;

    T& index(uint64_t offset);

    arma::Mat<T> &at(uint32_t channel);

    const arma::Mat<T> &at(uint32_t channel) const;

    void padding(const std::vector<uint32_t> &pads, T padding_value);

    void fill(const std::vector<T> &values); // values is row major

    void fill(T val);

    void ones();

    void rand();

    void show() const;

    void flatten();

    std::shared_ptr<Tensor<T>> clone();

    static std::shared_ptr<Tensor<T>> ElementAdd(const std::shared_ptr<Tensor<T>> &tensor1,
                                                 const std::shared_ptr<Tensor<T>> &tensor2);
    
    static std::shared_ptr<Tensor<T>> ElementMultiply(const std::shared_ptr<Tensor<T>> &tensor1,
                                                      const std::shared_ptr<Tensor<T>> &tensor2);

};


}

#endif