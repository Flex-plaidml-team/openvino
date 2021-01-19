// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/one_hot.hpp"

namespace LayerTestsDefinitions {

std::string OnehotLayerTest::getTestCaseName(testing::TestParamInfo<oneHotParams> obj) {
    int64_t axis;
    size_t depth;
    float onValue;
    float offValue;
    std::vector<size_t> indicesShape;
    InferenceEngine::Precision netPrecision;
    LayerTestsUtils::TargetDevice targetDevice;
    std::tie(axis, depth, onValue, offValue, indicesShape, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "axis=" << axis << "_";
    result << "depth=" << depth << "_";
    result << "onValue=" << onValue << "_";
    result << "offValue=" << offValue << "_";
    result << "IS=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void OnehotLayerTest::SetUp() {
    int64_t axis;
    size_t depth;
    float onValue;
    float offValue;
    std::vector<size_t> indicesShape;
    InferenceEngine::Precision netPrecision;
    std::tie(axis, depth, onValue, offValue, indicesShape, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {indicesShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

    const auto depthOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{}, depth);
    const auto onValueOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{}, onValue);
    const auto offValueOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{}, offValue);
    auto onehot = std::make_shared<ngraph::opset1::OneHot>(params.at(0), depthOp, onValueOp, offValueOp, axis);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(onehot)};
    function = std::make_shared<ngraph::Function>(results, params, "onehot");
}

}  // namespace LayerTestsDefinitions
