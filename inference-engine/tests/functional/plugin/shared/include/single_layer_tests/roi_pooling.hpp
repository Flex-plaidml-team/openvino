// Copyright (C) 2019-2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using ROIPoolingSpecificParams = std::tuple<
   std::string,       // specifies a method to perform pooling.
   size_t,            // ROI region height.
   size_t,            // ROI region width.
   float              // spatial_scale.
>;

using ROIPoolingParams = std::tuple<
    ROIPoolingSpecificParams,
    InferenceEngine::Precision,       // Net precision
    std::vector<size_t>,              // Input shape
    std::vector<std::vector<float>>,  // Input box tensor
    std::string                       // Device name
>;

class ROIPoolingLayerTest : public testing::WithParamInterface<ROIPoolingParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ROIPoolingParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
