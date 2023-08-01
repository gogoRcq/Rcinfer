#include "glog/logging.h"
#include "gtest/gtest.h"
#include <armadillo>
#include "data/Tensor.h"

TEST (test_tensor, create) {
    using namespace rq;
    Tensor<float> tensor(32, 32, 3);
    ASSERT_EQ(tensor.channels(), 3);
    ASSERT_EQ(tensor.rows(), 32);
    ASSERT_EQ(tensor.cols(), 32);
}

TEST(test_tensor, fill) {
	using namespace rq;
	Tensor<float> tensor(3, 3, 3);
	ASSERT_EQ(tensor.channels(), 3);
	ASSERT_EQ(tensor.rows(), 3);
	ASSERT_EQ(tensor.cols(), 3);

	std::vector<float> values;
	for (int i = 0; i < 27; ++i) {
		values.push_back((float) i);
	}
	tensor.fill(values);
	LOG(INFO) << "\n" << tensor.data();

	int index = 0;
	for (int c = 0; c < tensor.channels(); ++c) {
		for (int r = 0; r < tensor.rows(); ++r) {
			for (int c_ = 0; c_ < tensor.cols(); ++c_) {
				ASSERT_EQ(values.at(index), tensor.at(c, r, c_));
				index += 1;
			}
		}
	}
	LOG(INFO) << "Test1 passed!";
}

TEST(test_tensor, padding1) {
	using namespace rq;
	Tensor<float> tensor(3, 3, 3);
	ASSERT_EQ(tensor.channels(), 3);
	ASSERT_EQ(tensor.rows(), 3);
	ASSERT_EQ(tensor.cols(), 3);

	tensor.fill(1.f); // 填充为1
	tensor.padding({1, 1, 1, 1}, 0); // 边缘填充为0
	ASSERT_EQ(tensor.rows(), 5);
	ASSERT_EQ(tensor.cols(), 5);

	int index = 0;
	// 检查一下边缘被填充的行、列是否都是0
	for (int c = 0; c < tensor.channels(); ++c) {
		for (int r = 0; r < tensor.rows(); ++r) {
			for (int c_ = 0; c_ < tensor.cols(); ++c_) {
				if (c_ == 0 || r == 0) {
					ASSERT_EQ(tensor.at(c, r, c_), 0);
				}
				index += 1;
			}
		}
	}
	LOG(INFO) << "Test2 passed!";
}

TEST(test_tensor, reshape) {
	using namespace rq;
	Tensor<float> tensor(3, 3, 1);

	arma::fmat input_data = "71 22 63;"
							"12 13 99;"
							"52 15 19;";
	tensor.at(0) = input_data;
	LOG(INFO) << "origin data:";
	LOG(INFO) << tensor.data(); 
	auto tensor1 = tensor.clone();

	tensor.ReRawshape({9});
	LOG(INFO) << "current data:";
	LOG(INFO) << tensor.data(); 

	tensor1->ReRawView({9});
	LOG(INFO) << "current data:";
	LOG(INFO) << tensor1->data(); 
}

TEST(test_tensor, reshape_3d) {
	using namespace rq;
	Tensor<float> tensor(3, 3, 2);

	arma::fmat input_data = "71 22 63;"
							"12 13 99;"
							"52 15 19;";
	tensor.at(0) = input_data;
	tensor.at(1) = input_data;
	LOG(INFO) << "origin data:";
	LOG(INFO) << tensor.data(); 
	auto tensor1 = tensor.clone();

	tensor.ReRawshape({1, 3, 6});
	LOG(INFO) << "current data:";
	LOG(INFO) << tensor.data(); 

	tensor1->ReRawView({1, 3, 6});
	LOG(INFO) << "current data:";
	LOG(INFO) << tensor1->data(); 
}
