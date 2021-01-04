// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using squaredDifferenceParams =  std::tuple<
    InferenceEngine::Precision,     // Net precision
    std::vector<std::vector<size_t>>,   // Input shapes
    std::string                     // Target device name
>;

class SquaredDifferenceLayerTest : public testing::WithParamInterface<squaredDifferenceParams>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<squaredDifferenceParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions