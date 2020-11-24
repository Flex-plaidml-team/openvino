// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/read_value.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>
#include <string>

using LayerTestsDefinitions::ReadValueLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::I64
};


const std::vector<std::vector<size_t>> initShapes = {
    {2, 3}, {2, 3, 4}, {2, 3, 4, 5}        //
};

INSTANTIATE_TEST_CASE_P(ReadValueCheck, ReadValueLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),                          //
                                            ::testing::ValuesIn(initShapes),                            //
                                            ::testing::Values(std::string("id")),          //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),          //
                         ReadValueLayerTest::getTestCaseName);
}  // namespace
