// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/bucketize.hpp"

namespace LayerTestsDefinitions {
std::string BucketizeLayerTest::getTestCaseName(testing::TestParamInfo<bucketizeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::vector<size_t> buckets;
    InferenceEngine::Precision outputPrecision;
    bool withRightBound;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inputShapes, buckets, outputPrecision, withRightBound, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Bucket=" << CommonTestUtils::vec2str(buckets) << "_";
    result << "outputPrecision=" << outputPrecision.name() << "_";
    result << "withRightBound=" << withRightBound << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void BucketizeLayerTest::SetUp() {
    // Use IE ref mode as ngraph can not run this operation now
    SetRefMode(LayerTestsUtils::RefMode::IE);
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShapes;
    std::vector<size_t> buckets;
    InferenceEngine::Precision outputPrecision;
    bool withRightBound;
    std::tie(netPrecision, inputShapes, buckets, outputPrecision, withRightBound, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto outputPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outputPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes, buckets});
    auto paramIn = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto bucketize = std::dynamic_pointer_cast<ngraph::opset3::Bucketize>(
            std::make_shared<ngraph::opset3::Bucketize>(paramIn[0], paramIn[1], outputPrc, withRightBound));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(bucketize)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Bucketize");
}

}  // namespace LayerTestsDefinitions
