// Copyright (C) 2019-2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roi_pooling.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ie_core.hpp"

namespace LayerTestsDefinitions {

std::string ROIPoolingLayerTest::getTestCaseName(testing::TestParamInfo<ROIPoolingParams> obj) {
    ROIPoolingSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShapes;
    std::vector<int> boxShapes;
    std::string targetDevice;
    std::tie(poolParams, netPrecision, inputShapes, boxShapes, targetDevice) = obj.param;
    std::string method;
    size_t pooledHeight;
    size_t pooledWidth;
    float spatialScale;
    std::tie(method, pooledHeight, pooledWidth, spatialScale) = poolParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "ROIS" << CommonTestUtils::vec2str(boxShapes) << "_";
    result << "M" << method << "_";
    result << "PH" << pooledHeight << "_";
    result << "PW" << pooledWidth << "_";
    result << "Spatial=" << spatialScale << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ROIPoolingLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    ROIPoolingSpecificParams poolParams;
    std::vector<size_t> inputShapes;
    std::vector<std::vector<float>> coordsBox;
    InferenceEngine::Precision netPrecision;
    std::tie(poolParams, netPrecision, inputShapes, coordsBox, targetDevice) = this->GetParam();
    std::string method;
    size_t pooledHeight;
    size_t pooledWidth;
    float spatialScale;
    std::tie(method, pooledHeight, pooledWidth, spatialScale) = poolParams;

    ngraph::Shape POIShape = {pooledHeight, pooledWidth};
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});

    // flat coordsBox for building ngraph constant node.
    ngraph::Shape constShape = {coordsBox.size(), coordsBox[0].size()};
    std::vector<float> flatCoords;
    for (auto roi : coordsBox) {
        flatCoords.insert(flatCoords.end(), roi.begin(), roi.end());
    }
    auto ROINode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, constShape, flatCoords.data());

    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto ROIPooling = std::dynamic_pointer_cast<ngraph::opset4::ROIPooling>(
        std::make_shared<ngraph::opset4::ROIPooling>(paramOuts[0], ROINode, POIShape, spatialScale, method));

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(ROIPooling)};
    function = std::make_shared<ngraph::Function>(results, params, "ROIPooling");
}

TEST_P(ROIPoolingLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions
