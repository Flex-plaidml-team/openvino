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
    std::shared_ptr<ngraph::Variable>  variable;
    std::string variable_id;
    std::tie(netPrecision, newValue, variable, variable_id, targetDevice) = obj.param;
    //std::tie(netPrecision, newValue, variable_id, targetDevice) = obj.param;
    std::ostringstream result;
    result << "newValue=" << CommonTestUtils::vec2str(newValue) << "_";
    //result << "variable_info" << variable->get_info() << "_";
    result << "variable_id=" << variable_id << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void AssignLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    InferenceEngine::SizeVector newValue;
    std::shared_ptr<ngraph::Variable> variable;
    std::string variable_id;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, newValue, variable, variable_id, targetDevice) = this->GetParam();
    //std::tie(netPrecision, newValue, variable_id, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { newValue });
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto assign = std::dynamic_pointer_cast<ngraph::opset3::Assign>(
            std::make_shared<ngraph::opset3::Assign>(paramIn[0], variable_id));
    //auto assign = std::make_shared<ngraph::opset3::Assign>(paramIn[0], variable_id);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(assign) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Assign");
}

TEST_P(AssignLayerTest, CompareWithRefsDynamicBath) {
    Run();
}
}  // namespace LayerTestsDefinitions
