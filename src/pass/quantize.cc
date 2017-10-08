/*!
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief Quantize the graph to lowbit operator
 */
#include <tvm/runtime/registry.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <unordered_set>
#include <string>
#include <cmath>
#include "../compiler/graph_transform.h"

namespace nnvm {
namespace pass {
namespace {

using compiler::FQuantizedOp;
using compiler::FCalibrate;

Graph QuantizeGraph(nnvm::Graph&& src) {
  static auto& quantized_op_map = Op::GetAttr<FQuantizedOp>("FQuantizedOp");
  static auto& fcalibrate_map = Op::GetAttr<FCalibrate>("FCalibrate");
  const auto& base2_range = src.GetAttr<std::vector<int>>("base2_range");
  static constexpr float eps = 1e-04;
  const auto& idx = src.indexed_graph();
  std::unordered_map<Node*, NodeEntry> quantized_var;
  std::unordered_map<Node*, int> reverse_mirror;

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    if (quantized_op_map.count(n->op())) {
      std::unordered_map<std::string, std::string> dict;
      if (fcalibrate_map.count(n->op())) {
        auto fcalibrate = fcalibrate_map[n->op()];
        fcalibrate(nid, n, idx, base2_range, &dict);
      }

      NodePtr temp = MakeNode(n->op()->name.c_str(), n->attrs.name, n->inputs, n->attrs.dict).node;
      for (size_t i = 0; i < temp->inputs.size(); ++i) {
        const auto& e = temp->inputs[i];
        if (e.node->is_variable()) {
          if (quantized_var.count(e.node.get())) {
            n->inputs[i] = quantized_var.at(e.node.get());
          } else {
            int k = base2_range[idx.node_id(e.node.get())];
            float scale = float(std::pow(2, 7 - k)) * (1 - eps);

            NodeEntry mul = MakeNode("__mul_scalar__",
              "quantize_mul_" + e.node->attrs.name, {e}, {{"scalar", std::to_string(scale)}});
            NodeEntry cast = MakeNode("cast",
              "quantize_cast_" + e.node->attrs.name, {mul}, {{"dtype", "int8"}});
            quantized_var.emplace(e.node.get(), cast);
            temp->inputs[i] = cast;
          }
        }
      }

      auto fquantized_op = quantized_op_map[n->op()];
      NodePtr qnode = fquantized_op(temp, dict);
      reverse_mirror.emplace(qnode.get(), nid);

      std::vector<NodeEntry> outputs;
      outputs.reserve(qnode->num_outputs());
      for (uint32_t i = 0; i < qnode->num_outputs(); ++i) {
        outputs.emplace_back(NodeEntry{qnode, 0, i});
      }
      *ret = std::move(outputs);
      return true;
    } else {
      LOG(FATAL) << n->op()->name << " cannot be quantized yet.";
      return false;
    }
  };

  Graph ret = compiler::GraphTransform(std::move(src), transform);
  std::vector<NodeEntry> outputs;
  outputs.reserve(outputs.size());
  for (const auto& e : ret.outputs) {
    int k = base2_range[reverse_mirror.at(e.node.get())];
    float scale = float(std::pow(2, k)) / std::pow(2, 7) / (1 - eps);

    NodeEntry cast = MakeNode("cast",
      "dequantize_cast_" + e.node->attrs.name, {e}, {{"dtype", "float32"}});
    NodeEntry mul = MakeNode("__mul_scalar__",
      "dequantize_mul_" + e.node->attrs.name, {cast}, {{"scalar", std::to_string(scale)}});
    outputs.emplace_back(mul);
  }
  ret.outputs = outputs;
  return ret;
}

NNVM_REGISTER_PASS(Quantize)
.describe("")
.set_body(QuantizeGraph)
.set_change_graph(true);


Graph CollectInternalOutputs(Graph src) {
  std::vector<NodeEntry> outputs;
  outputs.reserve(src.indexed_graph().num_node_entries());
  DFSVisit(src.outputs, [&](const NodePtr& n) {
      for (uint32_t i = 0; i < n->num_outputs(); ++i) {
        outputs.emplace_back(NodeEntry{n, i, 0});
      }
    });

  Graph ret;
  ret.outputs = std::move(outputs);
  return ret;
}

TVM_REGISTER_GLOBAL("nnvm.quantization.CollectInternalOutputs")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
    *rv = CollectInternalOutputs(args[0]);
});

}  // namespace
}  // namespace pass
}  // namespace nnvm
