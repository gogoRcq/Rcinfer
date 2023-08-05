#include "details/softmax.h"
#include "common.h"
#include "runtime/StateCode.h"

namespace rq {

template<class T>
InferStatus rcSoftmaxLayer<T>::forwards(const std::vector<std::shared_ptr<Tensor<T>>> &inputs,
                                        std::vector<std::shared_ptr<Tensor<T>>> &outputs) {
    if (inputs.empty()) {
        LOG(ERROR) << "Input of relu is empty";
        return InferStatus::rInferFailedInputEmpty;
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::rInferFailedInputOutSizeAdaptingError;
    }

    const uint32_t batch_size = inputs.size();
    #pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<T>> &input = inputs.at(i);
        CHECK(input != nullptr && !input->empty()) << "The input feature map for softmax layer is empty";

        std::shared_ptr<Tensor<T>> output = outputs.at(i);
        if (output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<T>>(input->shapes()[0], input->shapes()[1], input->shapes()[2]);
            outputs.at(i) = output;
        }

        CHECK(input->shapes() == output->shapes()) << "The output size of softmax is error";

        const arma::Cube<T>& input_data = input->data();
        arma::Cube<T>& output_data = output->data();
        const T max = input_data.max(); 
        const T sum = arma::accu(arma::exp(input_data - max));
        const T offset = max + log(sum);

        output_data = arma::exp(input_data - offset);
    }
    return InferStatus::rInferSuccess;
}

INSTALLCLASS(rcSoftmaxLayer);

}