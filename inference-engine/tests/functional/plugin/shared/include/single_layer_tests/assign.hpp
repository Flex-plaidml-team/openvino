// Copyright (C) 2019 Intel Corporation
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
typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::vector<size_t>,                // New Value
        std::string,                        // variable_id
        std::string                         // Device name
> assignParams;

class AssignLayerTest : public testing::WithParamInterface<assignParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<assignParams> obj);

    protected:
        void SetUp() override;
};

}  // namespace LayerTestsDefinitions
