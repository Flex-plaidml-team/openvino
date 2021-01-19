// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

typedef std::tuple<InferenceEngine::SizeVector,  // Kernel size
                   InferenceEngine::SizeVector,  // Strides
                   std::vector<ptrdiff_t>,       // Pad begin
                   std::vector<ptrdiff_t>,       // Pad end
                   InferenceEngine::SizeVector,  // Dilation
                   size_t,                       // Num out channels
                   ngraph::op::PadType,          // Padding type
                   ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode,
                   float  // Padding Value
                   >
        binConvSpecificParams;
typedef std::tuple<binConvSpecificParams,
                   InferenceEngine::Precision,    // Net precision
                   InferenceEngine::SizeVector,   // Input shapes
                   LayerTestsUtils::TargetDevice  // Device name
                   >
        binConvLayerTestParamsSet;
namespace LayerTestsDefinitions {

class BinaryConvolutionLayerTest : public testing::WithParamInterface<binConvLayerTestParamsSet>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<binConvLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
