// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/deformable_convolution.hpp"

namespace LayerTestsDefinitions {

std::string DeformableConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<deformableConvLayerTestParamsSet> obj) {
    deformableConvSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector deformableShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inputShapes, deformableShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t group;
    size_t deformableGroup;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, group, deformableGroup, padType) = convParams;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "DS=" << CommonTestUtils::vec2str(deformableShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "G=" << group << "_";
    result << "DG=" << deformableGroup << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void DeformableConvolutionLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    deformableConvSpecificParams convParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> deformableShape;
    InferenceEngine::Precision netPrecision;
    std::tie(convParams, netPrecision, inputShape, deformableShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t group;
    size_t deformableGroup;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, group, deformableGroup, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, deformableShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> filterweights = {};
    auto shape = paramOuts[0].get_shape();
    std::vector<size_t> filterWeightsShape = { convOutChannels, shape[1] / group };
    filterWeightsShape.insert(filterWeightsShape.end(), kernel.begin(), kernel.end());
    auto filterWeightsNode = ngraph::builder::makeConstant(ngPrc, filterWeightsShape, filterweights, true);
    auto deformableConv = std::make_shared<ngraph::opset4::DeformableConvolution>(paramOuts[0], paramOuts[1],
            filterWeightsNode, stride, padBegin, padEnd, dilation, padType, group, deformableGroup);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deformableConv)};
    function = std::make_shared<ngraph::Function>(results, params, "deformableConvolution");
}

}  // namespace LayerTestsDefinitions