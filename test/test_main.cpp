#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("Rcinfer");
    FLAGS_logtostderr = true;
    FLAGS_log_dir = "./log/";
    return RUN_ALL_TESTS();
    return 0;
}
