// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/assign.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>
#include <string>

using LayerTestsDefinitions::AssignLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::I64
};

/*
const std::vector<std::vector<size_t>> newShapes = {
    {2, 3}, {2, 3, 4}, {2, 3, 4, 5}        //
};
*/
//ngraph::element::Type_t::f32
const ngraph::PartialShape data_shape = { 2, 3 };
//const ngraph::element::Type data_type =  InferenceEngine::Precision::FP32;
const ngraph::element::Type data_type = ngraph::element::Type_t::f32;
//std::string variable_id = "lstm";
//std::string variable_id;
const ngraph::VariableInfo info = { data_shape, data_type, "" };
const std::shared_ptr<ngraph::Variable> variable = std::make_shared<ngraph::Variable>(info);
INSTANTIATE_TEST_CASE_P(AssignCheck, AssignLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),                          //
                                            //::testing::ValuesIn(newShapes),                            //
                                            ::testing::Values(std::vector<size_t>({ 2, 3 })),  //
                                            ::testing::Values(variable),      //
                                            ::testing::Values(std::string("")),          //
                                            //::testing::Values(std::string()),
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),          //
                         AssignLayerTest::getTestCaseName);
}  // namespace
