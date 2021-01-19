// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<int64_t,                       // axis
                   size_t,                        // depth
                   float,                         // on_value
                   float,                         // off_value
                   std::vector<size_t>,           // Input shapes
                   InferenceEngine::Precision,    // Net precision
                   LayerTestsUtils::TargetDevice  // Target device name
                   >
        oneHotParams;

namespace LayerTestsDefinitions {

class OnehotLayerTest : public testing::WithParamInterface<oneHotParams>, public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<oneHotParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
