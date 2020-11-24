// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset3.hpp"

#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

#include "plaidml/edsl/edsl.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace ngraph;

namespace PlaidMLPlugin {

static OpRegistration reg("readvalue", [](const Context& ctx) {
  //auto* layer = ngraph::as_type<ngraph::opset4::ReadValue>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  //layer->validate_and_infer_types();
  //auto variable = layer->get_variable();
  //auto variable_info = variable->get_info();
  //std::vector<edsl::TensorDim> dims(I.rank());
  //I.bind_dims(dims);
  //std::vector<edsl::TensorIndex> I_idxs(I.rank());
  return edsl::make_tuple(I);
});

}  // namespace PlaidMLPlugin
