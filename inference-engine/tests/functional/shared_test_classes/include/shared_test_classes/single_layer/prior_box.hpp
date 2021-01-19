// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "details/ie_exception.hpp"
#include "ie_core.hpp"
#include "ie_precision.hpp"

#include "ngraph/opsets/opset1.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<std::vector<float>,  // min_size
                   std::vector<float>,  // max_size
                   std::vector<float>,  // aspect_ratio
                   std::vector<float>,  // density
                   std::vector<float>,  // fixed_ratio
                   std::vector<float>,  // fixed_size
                   bool,                // clip
                   bool,                // flip
                   float,               // step
                   float                // offset
                   >
        priorBoxAttrs;

typedef std::tuple<priorBoxAttrs,
                   std::vector<float>,                 // variance
                   bool,                               // scale_all_sizes
                   bool,                               // use fixed sizes
                   bool,                               // use fixed ratios
                   InferenceEngine::Precision,         // net precision
                   InferenceEngine::SizeVector,        // input shape
                   InferenceEngine::SizeVector,        // image shape
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
        priorBoxParams;

class PriorBoxLayerTest : public testing::WithParamInterface<priorBoxParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
