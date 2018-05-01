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
#include "./graph_transform.h"

namespace nnvm {
namespace compiler {

inline NodeEntry MakeReQtzNode(NodeEntry e, int threshold_bit) {
  std::string name = e.node->attrs.name;
  NodeEntry quantize = MakeNode("simulated_quantize",
    name + "_quantize", {e},
    {{"threshold_bit", std::to_string(threshold_bit)},
     {"num_bit", std::to_string(storage_bit)},
     {"out_type", "float32"}});
  return quantize;
}

Graph QuantizeGraph(nnvm::Graph&& src) {
  const IndexedGraph& idx = src.indexed_graph();
  static auto& fqtz_pattern = Op::GetAttr<TQtzPattern>("TQtzPattern");
  const auto& threshold = src.GetAttr<std::vector<int>>("threshold");

  std::vector<int> cnt(idx.num_node_entries(), 0);
  std::vector<bool> tagged(idx.num_node_entries(), false);
  DFSVisit(src.outputs, [&](const NodePtr& n) {
    if (!n->is_variable()) {
      for (const auto& e: n->inputs) {
        cnt[idx.entry_id(e)] ++;
        if (fqtz_pattern.count(e.node->op()) &&
            fqtz_pattern[e.node->op()] == kRequire)
          tagged[idx.entry_id(e)] = true;
      }
    }
  });

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    if (n->is_variable()) return false;
    std::vector<NodeEntry> outputs;
    outputs.reserve(n->num_outputs());

    for (size_t i = 0; i < n->num_outputs(); ++i) {
      NodeEntry e{n, 0, 0};
      uint32_t eid = idx.entry_id(e);
      if (n->is_variable() || cnt[eid] > 1 || tagged[eid]) {
        e = MakeReQtzNode(e, threshold[eid]);
      }
      outputs.emplace_back(std::move(e));
    }

    *ret = std::move(outputs);
    return true;
  };
  Graph ret = compiler::GraphTransform(src, transform);
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

}  // namespace compiler
}  // namespace nnvm
