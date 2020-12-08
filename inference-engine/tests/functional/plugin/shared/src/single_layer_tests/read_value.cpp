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
#include "single_layer_tests/read_value.hpp"

namespace LayerTestsDefinitions {
    std::string ReadValueLayerTest::getTestCaseName(testing::TestParamInfo<readvalueParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector initValue;
    std::string targetDevice;
    std::string variable_id;
    std::tie(netPrecision, initValue, variable_id, targetDevice) = obj.param;
    std::ostringstream result;
    result << "initValue=" << CommonTestUtils::vec2str(initValue) << "_";
    result << "variable_id=" << variable_id << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ReadValueLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    InferenceEngine::SizeVector initValue;
    std::string variable_id;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, initValue, variable_id, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, { initValue });
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto readvalue = std::make_shared<ngraph::opset3::ReadValue>(paramIn[0], variable_id);
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(readvalue) };
    function = std::make_shared<ngraph::Function>(results, paramsIn, "ReadValue");
}

TEST_P(ReadValueLayerTest, CompareWithRefsDynamicBath) {
    Run();
}
}  // namespace LayerTestsDefinitions
