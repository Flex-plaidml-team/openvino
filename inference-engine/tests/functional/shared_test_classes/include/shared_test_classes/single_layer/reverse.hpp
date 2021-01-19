// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

// Note: always give axes in "INDEX" mode, the test builder will convert to MASK mode when needed
typedef std::tuple<InferenceEngine::Precision,         // Network precision
                   std::vector<size_t>,                // Input shapes
                   std::vector<size_t>,                // Axes
                   std::string,                        // Mode
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
        reverseParams;

class ReverseLayerTest : public testing::WithParamInterface<reverseParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reverseParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
