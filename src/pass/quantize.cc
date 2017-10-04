/*!
 *  Copyright (c) 2016 by Contributors
 * \file quantization.cc
 * \brief Quantize the graph to lowbit operator
 */
#include <tvm/runtime/registry.h>
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/packed_func_ext.h>
#include <unordered_set>
#include "../compiler/graph_transform.h"

namespace nnvm {
namespace pass {
namespace {

Graph QuantizeGraph(nnvm::Graph&& src) {
  static auto& quantized_op_map = Op::GetAttr<FQuantizedOp>("FQuantizedOp");
  std::unordered_map<Node*, NodeEntry> mirror;

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;

    if (quantized_op_map.count(n->op())) {
      auto fquantized_op = quantized_op_map[n->op()];
      NodePtr qnode = fquantized_op(n);
      for (const auto& e : n->inputs) {
        NodeEntry input;
        if (e.node->is_variable()) {
          if (mirror.count(e.node.get())) {
            input = mirror.at(e.node.get());
          } else {
            // quantize variable
            NodeEntry mul = MakeNode("__mul_scalar__",
              "quantize_mul_" + e.node->attrs.name, {e}, {{"scalar", "1.0"}});
            NodeEntry cast = MakeNode("cast",
              "quantize_cast_" + e.node->attrs.name, {mul}, {{"dtype", "int8"}});
            mirror.emplace(e.node.get(), cast);
            input = cast;
          }
        } else {
          input = e;
        }
        qnode->inputs.emplace_back(std::move(input));
      }

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
  // dequantize outputs
  std::vector<NodeEntry> outputs;
  outputs.reserve(outputs.size());
  for (const auto& e : ret.outputs) {
    NodeEntry cast = MakeNode("cast",
      "dequantize_cast_" + e.node->attrs.name, {e}, {{"dtype", "float32"}});
    NodeEntry mul = MakeNode("__mul_scalar__",
      "dequantize_mul_" + e.node->attrs.name, {cast}, {{"scalar", "1.0"}});
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
