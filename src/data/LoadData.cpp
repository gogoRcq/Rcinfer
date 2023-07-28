#include "data/LoadData.h"
#include "common.h"

namespace rq {

template<class T>
std::pair<size_t, size_t> CSVDataLoader<T>::getMatrixSize(std::ifstream &file, char splitChar) {
    file.clear();
    size_t rows = 0, cols = 0;
    const auto startPos = file.tellg(); // get offset of read ptr

    std::string token;
    std::string line;
    std::stringstream line_stream;

    while (file.good()) {
        std::getline(file, line);
        if (line.empty()) break;

        line_stream.clear();
        line_stream.str(line);

        size_t line_cols = 0;
        while (line_stream.good()) {
            std::getline(line_stream, token, splitChar);
            ++line_cols;
        }
        cols = std::max(line_cols, cols);
        ++rows;
    }
    file.clear();
    file.seekg(startPos);
    return {rows, cols};
}

template<class T>
std::shared_ptr<Tensor<T>> CSVDataLoader<T>::loadData(const std::string &filePath, char splitChar) {
    CHECK(!filePath.empty()) << "file path is empty";
    std::ifstream inFile(filePath);
    CHECK(inFile.is_open() && inFile.good()) << "file open failed! " << filePath;

    const auto& [rows, cols] = CSVDataLoader::getMatrixSize(inFile, splitChar);
    std::shared_ptr<Tensor<T>> tensorPtr = std::make_shared<Tensor<T>>(rows, cols, 1);
    arma::Mat<T> &matData = tensorPtr->at(0);

    std::string token;
    std::string line;
    std::stringstream line_stream;

    size_t row = 0;
    while (inFile.good()) {
        std::getline(inFile, line);
        if (line.empty()) break;

        line_stream.clear();
        line_stream.str(line);
        
        size_t col = 0;
        while (line_stream.good()) {
            std::getline(line_stream, token, splitChar);
            double temp = std::stod(token);
            try {
                matData.at(row, col) = static_cast<T>(temp);
            } catch(const std::exception& e) {
                LOG(ERROR) << "Parse CSV File meet error: " << e.what();
            }
            ++col;
            CHECK(col <= cols) << "There are excessive elements on the column";
        }
        ++row;
        CHECK(row <= rows) << "There are excessive elements on the row";
    }
    return tensorPtr;
    inFile.close();
}

template<class T>
std::shared_ptr<Tensor<T>> CSVDataLoader<T>:: loadDataWithHeader(const std::string &filePath, 
                                                                 std::vector<std::string> &headers, 
                                                                 char splitChar) {
    CHECK(!filePath.empty()) << "file path is empty";
    std::ifstream inFile(filePath);
    CHECK(inFile.is_open() && inFile.good()) << "file open failed! " << filePath;

    const auto& [rows, cols] = CSVDataLoader::getMatrixSize(inFile, splitChar);
    CHECK(rows >= 1);
    std::shared_ptr<Tensor<T>> tensorPtr = std::make_shared<Tensor<T>>(rows - 1, cols, 1);
    arma::Mat<T> &matData = tensorPtr->at(0);

    std::string token;
    std::string line;
    std::stringstream line_stream;

    // read head;
    std::getline(inFile, line);
    line_stream.str(line);
    while (line_stream.good()) {
        std::getline(line_stream, token, splitChar);
        headers.emplace_back(token);
    }

    size_t row = 0;
    while (inFile.good()) {
        std::getline(inFile, line);
        if (line.empty()) break;

        line_stream.clear();
        line_stream.str(line);
        
        size_t col = 0;
        while (line_stream.good()) {
            std::getline(line_stream, token, splitChar);
            double temp = std::stod(token);
            try {
                matData.at(row, col) = static_cast<T>(temp);
            } catch(const std::exception& e) {
                LOG(ERROR) << "Parse CSV File meet error: " << e.what();
                continue;
            }
            ++col;
            CHECK(col <= cols) << "There are excessive elements on the column";
        }
        ++row;
        CHECK(row <= rows) << "There are excessive elements on the row";
    }
    inFile.close();
    return tensorPtr;
}

INSTALLCLASS(CSVDataLoader);

}