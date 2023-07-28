#include "data/Tensor.h"
#include "glog/logging.h"
#include <iostream>
#include "common.h"
#include <vector>

namespace rq {

template<class T>
Tensor<T>::Tensor(uint32_t row, uint32_t col, uint32_t channel):tsData(row, col, channel), rawShapes({row, col, channel}){}

template<class T>
Tensor<T>::Tensor(const Tensor<T>& tensor){
    this->tsData = tensor.tsData;
    this->rawShapes = tensor.rawShapes;
}

template<class T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T> &tensor){
    if (this != &tensor) {
        this->tsData = tensor.tsData;
        this->rawShapes = tensor.rawShapes;
    }
    return *this;
}

template<class T>
uint32_t Tensor<T>::cols() const {
    CHECK(!this->tsData.empty());
    return tsData.n_cols;
}

template<class T>
uint32_t Tensor<T>::rows() const {
    CHECK(!this->tsData.empty());
    return tsData.n_rows;
}

template<class T>
uint32_t Tensor<T>::channels() const {
    CHECK(!this->tsData.empty());
    return tsData.n_slices;
}

template<class T>
uint64_t Tensor<T>::size() const {
    CHECK(!this->tsData.empty());
    return tsData.size();
}

template<class T>
bool Tensor<T>::empty() const {
    return this->tsData.empty();
}

template<class T>
std::vector<uint32_t> Tensor<T>::shapes() const {
    CHECK(!this->tsData.empty());
    return {this->channels(), this->rows(), this->cols()};
}

template<class T>
arma::Cube<T>& Tensor<T>::data() {
    return this->tsData;
}

template<class T>
const arma::Cube<T>& Tensor<T>::data() const {
    return this->tsData;
}

template<class T>
void Tensor<T>::setdata(arma::Cube<T>& data) {
    CHECK(this->rows() == data.n_rows) << "the rows are not equal";
    CHECK(this->cols() == data.n_cols) << "the cols are not equal";
    CHECK(this->channels() == data.n_slices) << "the channels are not equal";
    this->tsData = data;
}

template<class T> 
void Tensor<T>::fill(T val){
    tsData.fill(val);
}

template<class T>
T Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK(!this->tsData.empty());
    return tsData.at(row, col, channel);
}

template<class T>
T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK(!this->tsData.empty());
    return tsData.at(row, col, channel);
}

template<class T>
T Tensor<T>::index(uint64_t offset) const {
    CHECK(!this->tsData.empty());
    return tsData.at(offset);
}

template<class T>
T& Tensor<T>::index(uint64_t offset){
    CHECK(!this->tsData.empty());
    return tsData.at(offset);
}

template<class T>
arma::Mat<T>& Tensor<T>::at(uint32_t channel) {
    CHECK_LT(channel, this->channels());
    return this->tsData.slice(channel);
}

template<class T>
const arma::Mat<T>& Tensor<T>::at(uint32_t channel) const {
    CHECK_LT(channel, this->channels());
    return this->tsData.slice(channel);
}

template<class T>
void Tensor<T>::fill(const std::vector<T> &values) {
    CHECK(!this->tsData.empty());
    CHECK_EQ(values.size(), this->tsData.size());
    
    const uint32_t row = this->tsData.n_rows;
    const uint32_t col = this->tsData.n_cols;
    const uint32_t channel = this->tsData.n_slices;
    const uint64_t planes = row * col;

    for (uint32_t i = 0; i < channel; i++) {
        arma::Mat<T> &channelData = this->tsData.slice(i);
        arma::Mat<T> channelDataTemp(values.data() + i * planes, col, row);
        channelData = channelDataTemp.t();
    }
}

template<class T>
void Tensor<T>::padding(const std::vector<uint32_t> &pads, T padding_value) {
    CHECK(!this->tsData.empty());
    CHECK_EQ(pads.size(), 4);

    const uint32_t upPad = pads[0];
    const uint32_t btPad = pads[1];
    const uint32_t ltPad = pads[2];
    const uint32_t rtPad = pads[3];
    const uint32_t channel = this->tsData.n_slices;

    arma::Cube<T> upCube(upPad, this->tsData.n_cols, channel);
    arma::Cube<T> btCube(btPad, this->tsData.n_cols, channel);
    arma::Cube<T> ltCube(this->tsData.n_rows + upPad + btPad, ltPad, channel);
    arma::Cube<T> rtCube(this->tsData.n_rows + upPad + btPad, rtPad, channel);
    upCube.fill(padding_value);
    btCube.fill(padding_value);
    ltCube.fill(padding_value);
    rtCube.fill(padding_value);
    
    this->tsData.insert_rows(0, upCube);
    this->tsData.insert_rows(this->tsData.n_rows, btCube);
    this->tsData.insert_cols(0, ltCube);
    this->tsData.insert_cols(this->tsData.n_cols, rtCube);
}

template<class T>
void Tensor<T>::ones() {
    CHECK(!this->tsData.empty());
    this->tsData.fill(1.);
}

template<class T>
void Tensor<T>::rand() {
    CHECK(!this->tsData.empty());
    this->tsData.randn();
}

template<class T>
void Tensor<T>::show() const {
    CHECK(!this->tsData.empty());
    for (uint32_t i = 0; i < this->tsData.n_slices; ++i) {
        LOG(INFO) << "channel: " << i;
        LOG(INFO)  << "\n" << this->tsData.slice(i);
    }
}

template<class T>
void Tensor<T>::flatten() {
    arma::Cube<T> tensorTemp(this->size(), 1, 1);

    const uint32_t row = this->tsData.n_rows;
    const uint32_t col = this->tsData.n_cols;
    const uint32_t channel = this->tsData.n_slices;
    uint64_t index = 0;

    // first trans then neno instruction optimization ？
    for (uint32_t ch = 0; ch < channel; ++ch) {
        arma::Mat<T>& channelData = this->tsData.slice(ch);
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                tensorTemp.at(index, 0, 0) = channelData.at(i, j);
                ++index;
            }
        }
    }

    CHECK_EQ(index, this->tsData.size());
    this->tsData = tensorTemp;
    this->rawShapes = {(uint32_t)index};
}

template<class T>
std::shared_ptr<Tensor<T>> Tensor<T>::clone() {
    return std::make_shared<Tensor<T>>(*this);
}

INSTALLCLASS(Tensor);

}


