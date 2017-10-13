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

static constexpr float eps = 1e-04;

inline NodeEntry MakeQuantizeNode(NodeEntry e, float s) {
  NodeEntry quantize = MakeNode("quantize",
    "quantize_" + e.node->attrs.name, {e}, {{"scale", std::to_string(s)}});
  return quantize;
}

inline NodeEntry MakeDequantizeNode(NodeEntry e, float s) {
  NodeEntry dequantize = MakeNode("dequantize",
    "dequantize_" + e.node->attrs.name, {e}, {{"scale", std::to_string(s)}});
  return dequantize;
}


Graph QuantizeGraph(nnvm::Graph&& src) {
  static auto& quantized_op_map = Op::GetAttr<FQuantizedOp>("FQuantizedOp");
  static auto& fcalibrate_map = Op::GetAttr<FCalibrate>("FCalibrate");
  // const auto& base2_range = src.GetAttr<std::vector<int>>("base2_range");
  const auto& scales_str = src.GetAttr<std::vector<std::string>>("scales");
  std::vector<float> scales;
  scales.reserve(scales_str.size());
  for (const auto& s : scales_str) {
    scales.emplace_back(std::stof(s));
  }
  int debug = src.GetAttr<int>("debug");
  const auto& idx = src.indexed_graph();
  std::unordered_map<Node*, NodeEntry> quantized_var;
  std::unordered_map<Node*, int> reverse_mirror;

  std::vector<NodeEntry> debug_outputs;
  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    if (quantized_op_map.count(n->op())) {
      std::unordered_map<std::string, std::string> dict;
      if (fcalibrate_map.count(n->op())) {
        auto fcalibrate = fcalibrate_map[n->op()];
        fcalibrate(nid, n, idx, scales, &dict);
      }

      NodePtr temp = MakeNode(n->op()->name.c_str(), n->attrs.name, n->inputs, n->attrs.dict).node;
      for (size_t i = 0; i < temp->inputs.size(); ++i) {
        const auto& e = temp->inputs[i];
        if (e.node->is_variable()) {
          if (quantized_var.count(e.node.get())) {
            n->inputs[i] = quantized_var.at(e.node.get());
          } else {
            float s = scales[idx.node_id(e.node.get())];
            NodeEntry quantize = MakeQuantizeNode(e, s);
            quantized_var.emplace(e.node.get(), quantize);
            temp->inputs[i] = quantize;
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
        if (debug) {
          debug_outputs.emplace_back(NodeEntry{qnode, 0, i});
        }
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
  const std::vector<NodeEntry>& src_outputs = debug ? debug_outputs : ret.outputs;
  outputs.reserve(src_outputs.size());
  for (const auto& e : src_outputs) {
    float s = scales[reverse_mirror.at(e.node.get())];
    NodeEntry dequantize = MakeDequantizeNode(e, s);
    outputs.emplace_back(dequantize);
  }
  ret.outputs = std::move(outputs);
  return ret;
}

NNVM_REGISTER_PASS(Quantize)
.describe("")
.set_body(QuantizeGraph)
.set_change_graph(true);


Graph CollectInternalOutputs(Graph src, bool include_vars=true) {
  std::vector<NodeEntry> outputs;
  outputs.reserve(src.indexed_graph().num_node_entries());
  DFSVisit(src.outputs, [&](const NodePtr& n) {
      if (!include_vars && n->is_variable()) return;
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
  if (args.size() == 1) {
    *rv = CollectInternalOutputs(args[0]);
  } else {
    *rv = CollectInternalOutputs(args[0], args[1]);
  }
});

}  // namespace
}  // namespace pass
}  // namespace nnvm
