#include "layer/ConvLayer.h"
#include "data/Tensor.h"
#include "glog/logging.h"
#include "operator/Operator.h"
#include <_types/_uint32_t.h>
#include <cstring>
#include <memory>
#include <sys/_types/_int32_t.h>
#include <sys/types.h>
#include <vector>

namespace rq {

template<class T>
std::shared_ptr<Layer<T>> ConvLayer<T>::creatorInstance(const std::shared_ptr<Operator> &op) {
    return std::make_shared<ConvLayer<T>>(op);
}

template<class T>
void ConvLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs, 
                            std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    CHECK(!inputs.empty());
    CHECK(op != nullptr);
    CHECK(op->opType == OperatorType::rOperatorConv);
    CHECK(inputs.size() == outputs.size());
    const std::vector<std::shared_ptr<Tensor<T>>>& weights = op->weights();
    CHECK(!weights.empty());

    std::vector<std::shared_ptr<Tensor<T>>> bias;
    if (op->isUseBias()) {
        bias = op->bias();
    }
    bool useBias = this->op->isUseBias();
    const uint32_t paddingH = op->getPaddingH();
    const uint32_t paddingW = op->getPaddingW();
    const uint32_t strideH = op->getStrideH();
    const uint32_t strideW = op->getStrideW();
    const uint32_t groups = op->getGroups();
    const uint32_t batchSize = inputs.size();

    for (uint32_t i = 0; i < batchSize; ++i) {
        const std::shared_ptr<Tensor<T>>& input = inputs[i];
        CHECK(input != nullptr && !input->empty());

        std::shared_ptr<Tensor<T>> input_;
        if (paddingH > 0 || paddingW > 0) {
            input_ = input->clone();
            input_->padding({paddingH, paddingH, paddingW, paddingW}, (T)0.0f);
        } else {
            input_ = input;
        }
        const uint32_t input_channels = input_->channels();
        const uint32_t input_rows = input_->rows();
        const uint32_t input_cols = input_->cols();
        const uint32_t output_channels = input_channels;
        const uint32_t kernel_counts = weights.size();
        const uint32_t kernel_rows = weights[0]->rows();
        const uint32_t kernel_cols = weights[0]->cols();
        CHECK(kernel_cols > 0 && kernel_rows > 0);
        const uint32_t output_rows = (uint32_t)std::floor((input_rows - kernel_rows) / strideH + 1);
        const uint32_t output_cols = (uint32_t)std::floor((input_cols - kernel_cols) / strideW + 1);
        if (groups != 1) {
            CHECK(kernel_counts % groups == 0);
            CHECK(input_channels % groups == 0);
        }
        for (const auto& weight : weights) {
            CHECK(weight->rows() == kernel_rows);
            CHECK(weight->cols() == kernel_cols);
            CHECK(weight->channels() == input_channels / groups);
        }
        uint32_t row_len = kernel_cols * kernel_cols; // kernel展开的长度
        uint32_t col_len = output_rows * output_cols; // 一行的长度，其实也就是一个结果的一个 channel
        uint32_t input_group_channels = input_channels / groups; // 每组的input的 channel 数
        uint32_t kernel_count_group = kernel_counts / groups; // 每组有多少个 kernel
        
        for (int g = 0; g < groups; ++g) {
            std::vector<arma::Mat<T>> kernel_mat_arr(kernel_count_group);
            arma::Mat<T> temp(1, row_len *  input_group_channels);
            for (int k = 0; k < kernel_count_group; ++k) {
                const std::shared_ptr<Tensor<T>>& kernel = weights[k + g * kernel_count_group];
                //std::memcpy(temp.memptr(), kernel->data().memptr(), sizeof(T) * row_len * input_group_channels);
                for (int igc = 0; igc < input_group_channels; ++igc) {
                    std::memcpy(temp.memptr() + row_len * igc, kernel->at(igc).memptr(), row_len * sizeof(T));
                }
                LOG(INFO) << "kernel展开后: " << "\n" << temp;
                kernel_mat_arr[k] = temp;
            }
            arma::Mat<T> input_mat(row_len * input_group_channels, col_len);
            for (int igc = 0; igc < input_group_channels; ++igc) {
                const arma::Mat<T>& channel = input_->at(igc + g * input_group_channels);
                uint32_t cur_col = 0;
                for (uint32_t col = 0; col < output_cols; ++col) {
                    for (uint32_t row = 0; row < output_rows; ++row) {
                        T* input_mat_ptr = input_mat.colptr(cur_col) + igc * row_len;
                        ++cur_col;
                        for (int kl = 0; kl < kernel_cols; ++kl) {
                            memcpy(input_mat_ptr + kl * kernel_rows, channel.colptr(kl + col * strideW) + row * strideH, sizeof(T) * kernel_rows);
                        }
                    }
                }
            }
            LOG(INFO)  << "input展开后: " << "\n"  << input_mat;

            std::shared_ptr<Tensor<T>> output = outputs[i];
            if (output == nullptr || output->empty()) {
                output = std::make_shared<Tensor<T>>(output_rows, output_cols, kernel_counts);
                outputs[i] = output;
            }

            std::vector<arma::Mat<T>> output_mats(kernel_count_group);
            for (int om = 0; om < kernel_count_group; ++om) {
                output_mats[om] = kernel_mat_arr[om] * input_mat;
            }
            for (int k = 0; k < kernel_count_group; ++k) {
                std::shared_ptr<Tensor<T>> bias_;
                if (!bias.empty() && useBias) {
                    bias_ = bias[k];
                }
                arma::Mat<T> output_mat = output_mats[k];
                CHECK(output_mat.size() == output_cols * output_rows);
                output_mat.reshape(output_rows, output_cols);
                if (!bias.empty() && useBias) {
                    T bias_val = bias_->index(0);
                    output_mat += bias_val;
                }
                output->at(k + g * kernel_count_group) = std::move(output_mat);
            }
        }
    }
}

INSTALLCLASS(ConvLayer);
REGISTER_CREATOR(conv, OperatorType::rOperatorConv, ConvLayer);

}