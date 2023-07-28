//
// Created by fss on 22-11-12.
//

#ifndef STATUS_CODE_H_
#define STATUS_CODE_H_

namespace rq {

enum class RuntimeParamType {
  rParameterUnknown = 0,
  rParameterBool = 1,
  rParameterInt = 2,
  rParameterFloat = 3,
  rParameterString = 4,
  rParameterIntArray = 5,
  rParameterFloatArray = 6,
  rParameterStringArray = 7,
};

enum class InferStatus {
  rInferUnknown = -1,
  rInferFailedInputEmpty = 1,
  rInferFailedWeightParameterError = 2,
  rInferFailedBiasParameterError = 3,
  rInferFailedStrideParameterError = 4,
  rInferFailedDimensionParameterError = 5,
  rInferFailedChannelParameterError = 6,
  rInferFailedInputOutSizeAdaptingError = 6,

  rInferFailedOutputSizeError = 7,
  rInferFailedOperationUnknown = 8,
  rInferFailedYoloStageNumberError = 9,

  rInferSuccess = 0,
};

enum class ParseParamAttrStatus {
  rParameterMissingUnknown = -1,
  rParameterMissingStride = 1,
  rParameterMissingPadding = 2,
  rParameterMissingKernel = 3,
  rParameterMissingUseBias = 4,
  rParameterMissingInChannel = 5,
  rParameterMissingOutChannel = 6,

  rParameterMissingEps = 7,
  rParameterMissingNumFeatures = 8,
  rParameterMissingDim = 9,
  rParameterMissingExpr = 10,
  rParameterMissingOutHW = 11,
  rParameterMissingShape = 12,
  rParameterMissingGroups = 13,
  rParameterMissingScale = 14,
  rParameterMissingResizeMode = 15,

  rAttrMissingBias = 21,
  rAttrMissingWeight = 22,
  rAttrMissingRunningMean = 23,
  rAttrMissingRunningVar = 24,
  rAttrMissingOutFeatures = 25,
  rAttrMissingYoloStrides = 26,
  rAttrMissingYoloAnchorGrides = 27,
  rAttrMissingYoloGrides = 28,

  rParameterAttrParseSuccess = 0
};
}
#endif //KUIPER_COURSE_INCLUDE_COMMON_HPP_
