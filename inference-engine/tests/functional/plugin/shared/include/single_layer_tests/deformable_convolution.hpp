// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

// ! [test_convolution:definition]
typedef std::tuple<
        InferenceEngine::SizeVector,    // Kernel size
        InferenceEngine::SizeVector,    // Strides
        std::vector<ptrdiff_t>,         // Pad begin
        std::vector<ptrdiff_t>,         // Pad end
        InferenceEngine::SizeVector,    // Dilation
        size_t,                         // Num out channels
        ngraph::op::PadType,            // Padding type
        int64_t,                        // Group
        int64_t                         // Deformable_group
> deformableConvSpecificParams;
typedef std::tuple<
        deformableConvSpecificParams,
        InferenceEngine::Precision,     // Net precision
        InferenceEngine::SizeVector,    // Input shapes
        InferenceEngine::SizeVector,    // Deformable shape
        LayerTestsUtils::TargetDevice   // Device name
> deformableConvLayerTestParamsSet;
namespace LayerTestsDefinitions {


class DeformableConvolutionLayerTest : public testing::WithParamInterface<deformableConvLayerTestParamsSet>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<deformableConvLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};
// ! [test_deformableconvolution:definition]

}  // namespace LayerTestsDefinitions