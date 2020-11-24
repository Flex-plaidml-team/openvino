// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset3.hpp"

#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

#include "plaidml/edsl/edsl.h"

#include <iostream>
#include <typeinfo>
#include "ngraph/op/assign.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/op/read_value.hpp"
using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace ngraph;

namespace PlaidMLPlugin {

static OpRegistration reg("assign", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset3::Assign>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  //layer->validate_and_infer_types();
  auto variable = layer->get_variable();
  auto variable_info = variable->get_info();
  auto variable_id = layer->get_variable_id();
  NodeVector start_nodes;
  for (const auto& input : layer->inputs()) {
    start_nodes.push_back(input.get_source_output().get_node_shared_ptr());
  }
  auto nodes = topological_sort(start_nodes);
  for (const auto& node : nodes) {
    if (auto read_value = as_type_ptr<ngraph::op::v3::ReadValue>(node)) {
      if (read_value->get_variable_id() == variable_id) {
        auto assign_input = layer->input(0);
        auto assign_output = assign_input.get_source_output();
        //read_value->set_argument(0, assign_output); //this will cause 'killed' error
      }
    }
  }
  /*
  std::vector<edsl::TensorDim> dims(I.rank());
  I.bind_dims(dims);
  std::vector<edsl::TensorIndex> I_idxs(I.rank());
  return edsl::make_tuple(edsl::Contraction().outShape(dims).outAccess(I_idxs).assign(I(I_idxs)));
  */
  return edsl::make_tuple(I);
});

}  // namespace PlaidMLPlugin
