#include "layer/MaxPoolingLayer.h"
#include "common.h"

namespace rq {

template<class T>
void MaxPoolingLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    CHECK(this->op != NULL);
    CHECK(this->op->opType == OperatorType::rOperatorMaxPooling);
    CHECK(!inputs.empty());

    uint32_t poolingH = this->op->getPoolingH();
    uint32_t poolingW = this->op->getPoolingW();
    uint32_t strideH = this->op->getStrideH();
    uint32_t strideW = this->op->getStrideW();
    uint32_t paddingH = this->op->getPaddingH();
    uint32_t paddingW = this->op->getPaddingW();
    uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; i++){
        const std::shared_ptr<Tensor<T>>& input_data = inputs.at(i)->clone();
        input_data->padding({paddingH, paddingH, paddingW, paddingW}, std::numeric_limits<T>::lowest());
        const uint32_t input_rows = input_data->rows();
        const uint32_t input_cols = input_data->cols();
        const uint32_t input_channels = input_data->channels();
        const uint32_t output_channels = input_channels;
        const uint32_t output_rows = (uint32_t)std::floor((input_rows - poolingH) / strideH + 1);
        const uint32_t output_cols = (uint32_t)std::floor((input_cols - poolingW) / strideW + 1);
        std::shared_ptr<Tensor<T>> output_data = std::make_shared<Tensor<T>>(output_rows, output_cols, output_channels);
        CHECK(output_rows > 0 && output_cols > 0);

        for (uint32_t ch = 0; ch < output_channels; ++ch) {
            const arma::Mat<T>& in_channel = input_data->at(ch);
            arma::Mat<T>& out_channel = output_data->at(ch);
            uint32_t up_row = 0, left_col = 0, down_row = poolingH - 1, right_col = poolingW - 1;
            for (uint32_t row = 0; row < output_rows; ++row) {
                for (uint32_t col = 0; col < output_cols; ++col) {
                    const arma::Mat<T>& subM = in_channel.submat(up_row + row * strideH, left_col + col * strideW, 
                                                                 down_row + row * strideH, right_col + col * strideW);
                    out_channel.at(row, col) = subM.max();
                }
            }
        }
        outputs.push_back(output_data);
    }                            
}

template<class T>
std::shared_ptr<Layer<T>> MaxPoolingLayer<T>::creatorInstance(const std::shared_ptr<Operator>& op) {
    return std::make_shared<MaxPoolingLayer<T>>(op);
}

INSTALLCLASS(MaxPoolingLayer);
REGISTER_CREATOR(maxpooling, OperatorType::rOperatorMaxPooling, MaxPoolingLayer);
  
} // namespace rq
