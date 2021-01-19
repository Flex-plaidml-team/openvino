// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<InferenceEngine::Precision,         // Network precision
                   std::vector<size_t>,                // Input data, N-D tensor
                   std::vector<size_t>,                // Buckets, 1-D tensor
                   InferenceEngine::Precision,         // output precision
                   bool,                               // with right bound
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
        bucketizeParams;

class BucketizeLayerTest : public testing::WithParamInterface<bucketizeParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<bucketizeParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
