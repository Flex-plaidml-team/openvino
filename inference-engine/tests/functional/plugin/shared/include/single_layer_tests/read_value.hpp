// Copyright (C) 2019 Intel Corporatio
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
        std::vector<size_t>,                // Init Value
        std::string,                        // variable_id
        std::string                         // Device name
> readvalueParams;

class ReadValueLayerTest : public testing::WithParamInterface<readvalueParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<readvalueParams> obj);

    protected:
        void SetUp() override;
};

}  // namespace LayerTestsDefinitions
