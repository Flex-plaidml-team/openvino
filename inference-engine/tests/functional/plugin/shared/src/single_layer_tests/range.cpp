// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_precision.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/range.hpp"

namespace LayerTestsDefinitions {

std::string RangeLayerTest::getTestCaseName(testing::TestParamInfo<RangeParams> obj) {
    InferenceEngine::Precision netPrecision;
    float start, stop, step;
    std::string targetDevice;
    std::tie(start, stop, step, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "Start=" << start << separator;
    result << "Stop=" << stop << separator;
    result << "Step=" << step << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void RangeLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    InferenceEngine::Precision netPrecision;
    std::tie(start, stop, step, netPrecision, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params_start = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{}, start);
    auto params_stop = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{}, stop);
    auto params_step = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{}, step);

    auto range = std::make_shared<ngraph::opset3::Range>(params_start, params_stop, params_step);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(range)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{}, "Range");
}

TEST_P(RangeLayerTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions