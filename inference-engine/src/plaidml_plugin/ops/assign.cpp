// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

#include "plaidml/edsl/edsl.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace ngraph;

namespace PlaidMLPlugin {

static OpRegistration reg("assign", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset3::Assign>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  layer->validate_and_infer_types();
  auto variable = layer->get_variable();
  auto variable_info = variable->get_info();
  std::vector<edsl::TensorDim> dims(I.rank());
  I.bind_dims(dims);
  std::vector<edsl::TensorIndex> I_idxs(I.rank());
  //if (layer->get_variable_id == variable_info.variable_id) {
  return edsl::make_tuple(edsl::Contraction().outShape(dims).outAccess(I_idxs).assign(I(I_idxs)));
  //} else{
    //THROW_IE_EXCEPTION << "Variables identifiers are inconsistent.";
  //}
});

}  // namespace PlaidMLPlugin
