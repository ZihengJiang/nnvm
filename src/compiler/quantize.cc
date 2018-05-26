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

TVM_REGISTER_GLOBAL("nnvm.quantization.SetQuantizeConfig")
.set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue *rv) {
  QuantizeConfigThreadLocalEntry *entry = QuantizeConfigThreadLocalStore::Get();
  entry->storage_bit = args[0];
  entry->accumulate_bit = args[1];
  entry->storage_dtype = args[2].operator std::string();
  entry->accumulate_dtype = args[3].operator std::string();
});



inline NodeEntry MakeReQtzNode(NodeEntry e, int threshold_bit) {
  auto *entry = QuantizeConfigThreadLocalStore::Get();
  std::string name = e.node->attrs.name;
  double threshold = std::pow(2, double(threshold_bit));
  NodeEntry clipped_data = MakeNode("clip",
    name + "_clipped", {e},
    {{"a_min", std::to_string(-threshold)},
     {"a_max", std::to_string(threshold)}});

  double scale = (std::pow(2, double(entry->storage_bit - 1)) - 1) / threshold;
  NodeEntry scaled_data = MakeNode("__mul_scalar__",
    name + "_scaled", {clipped_data},
    {{"scalar", std::to_string(scale)}});

  NodeEntry rounded_data = MakeNode("around",
    name + "_rounded", {scaled_data},
    {{"mode", "symmetry"}});

  NodeEntry rescaled_data = MakeNode("__div_scalar__",
    name + "_rescaled", {rounded_data},
    {{"scalar", std::to_string(scale)}});

  return rescaled_data;
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
        if (fqtz_pattern.count(n->op()) &&
            fqtz_pattern[n->op()] == kRequire) {
          uint32_t eid = idx.entry_id(e);
          tagged[eid] = true;
        }
      }
    }
  });

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
    std::vector<NodeEntry> outputs;
    outputs.reserve(n->num_outputs());

    for (size_t i = 0; i < n->num_outputs(); ++i) {
      NodeEntry e{n, i, 0};
      uint32_t eid = idx.entry_id(nid, i);
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
