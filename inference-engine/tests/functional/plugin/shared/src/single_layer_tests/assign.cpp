// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ie_plugin_config.hpp>
#include <ie_core.hpp>
#include <functional>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "single_layer_tests/assign.hpp"

namespace LayerTestsDefinitions {
    std::string AssignLayerTest::getTestCaseName(testing::TestParamInfo<assignParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector newValue;
    std::string targetDevice;
    std::string variable_id;
    std::tie(netPrecision, newValue, variable_id, targetDevice) = obj.param;
    std::ostringstream result;
    result << "newValue=" << CommonTestUtils::vec2str(newValue) << "_";
    result << "variable_id=" << variable_id << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AssignLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    InferenceEngine::SizeVector newValue;
    std::string variable_id;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, newValue, variable_id, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    //FYI:https://github.com/Flex-plaidml-team/openvino/blob/plaidml/inference-engine/tests/functional/plugin/shared/src/execution_graph_tests/keep_assing.cpp
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { newValue });
    auto init_constant = std::make_shared<ngraph::op::v0::Constant>(ngPrc, newValue, 0);
    auto read_value = std::make_shared<ngraph::opset4::ReadValue>(init_constant, "id");
    auto mul = std::make_shared<ngraph::op::v1::Multiply>(paramsIn[0], read_value);
    auto assign = std::make_shared<ngraph::opset3::Assign>(mul, variable_id);
    assign->add_control_dependency(read_value);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(assign)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Assign");
}

TEST_P(AssignLayerTest, CompareWithRefsDynamicBath) {
    Run();
}
}  // namespace LayerTestsDefinitions
