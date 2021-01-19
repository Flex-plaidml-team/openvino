// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reverse.hpp"

namespace LayerTestsDefinitions {
std::string ReverseLayerTest::getTestCaseName(testing::TestParamInfo<reverseParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::vector<size_t> axes;
    std::string mode, targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inputShapes, axes, mode, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "AX=" << CommonTestUtils::vec2str(axes) << "_";
    result << "mode=" << mode << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ReverseLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    std::vector<size_t> axes;
    std::string mode;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShapes, axes, mode, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    std::shared_ptr<ngraph::opset1::Constant> constNode;
    auto axes_dtype = (mode == "index") ? ngraph::element::Type_t::i64 : ngraph::element::Type_t::boolean;
    constNode = std::make_shared<ngraph::opset1::Constant>(axes_dtype, ngraph::Shape{axes.size()}, axes);
    auto reverse = std::dynamic_pointer_cast<ngraph::opset1::Reverse>(
            std::make_shared<ngraph::opset1::Reverse>(paramIn[0], constNode, mode));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reverse)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Reverse");
}

}  // namespace LayerTestsDefinitions
