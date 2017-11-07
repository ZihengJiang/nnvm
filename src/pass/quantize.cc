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
#include <map>
#include "../compiler/graph_transform.h"

namespace nnvm {
namespace pass {
namespace {

using compiler::TCalibInfo;
using compiler::TScaleMap;
using compiler::FRTQuantize;

inline std::vector<NodeEntry> GetOutputs(NodePtr n) {
  std::vector<NodeEntry> outputs;
  outputs.reserve(n->num_outputs());
  for (uint32_t i = 0; i < n->num_outputs(); ++i) {
    outputs.emplace_back(NodeEntry{n, i, 0});
  }
  return outputs;
}

inline NodeEntry MakeQuantizeNode(NodeEntry e, NodeEntry s) {
  NodeEntry quantize = MakeNode("quantize",
    e.node->attrs.name + "_quantized", {e, s});
  return quantize;
}

inline NodeEntry MakeDequantizeNode(NodeEntry e, NodeEntry s) {
  NodeEntry dequantize = MakeNode("dequantize",
    e.node->attrs.name + "_dequantized", {e, s});
  return dequantize;
}

TCalibInfo CalibStatisticalAnalyse(const IndexedGraph& idx,
                                   uint32_t num_samples,
                                   const std::vector<int>& bounds) {
  CHECK_EQ(bounds.size(), num_samples * idx.num_node_entries());
  TCalibInfo info;
  info.reserve(idx.num_nodes());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    std::map<std::vector<int>, std::vector<int>> bound_map;

    const auto& inputs = idx[nid].inputs;
    uint32_t num_outputs = idx[nid].source->num_outputs();
    // assume only one output
    CHECK_EQ(num_outputs, 1);

    for (uint32_t i = 0; i < num_samples; ++i) {
      std::vector<int> ibounds;
      for (const auto& e : inputs) {
        ibounds.push_back(bounds[idx.entry_id(e) * num_samples + i]);
      }
      std::vector<int> obounds;
      for (uint32_t index = 0; index < num_outputs; ++index) {
        obounds.push_back(bounds[idx.entry_id(nid, index) * num_samples + i]);
      }
      if (bound_map.count(ibounds)) {
        const std::vector<int>& old = bound_map.at(ibounds);
        if (obounds[0] > old[0]) {
          bound_map[ibounds] = obounds;
        }
      } else {
        bound_map[ibounds] = obounds;
      }
    }

    // LOG(INFO) << idx[nid].source->attrs.name;
    // for (const auto& kv : bound_map) {
    //   std::cout << "inputs: ";
    //   for (auto bound : kv.first) std::cout << bound << " ";
    //   std::cout << ", outputs: ";
    //   for (auto bound : kv.second) std::cout << bound << " ";
    //   std::cout << std::endl;
    // }
    info.emplace_back(bound_map);
  }
  return info;
}

using compiler::TQuantizeConfig;

Graph QuantizeGraphWithoutCalibration(nnvm::Graph&& src) {
  LOG(INFO) << "quantize begin";
  const auto& idx = src.indexed_graph();
  LOG(INFO) << "num node: " << idx.num_nodes();
  static auto& quantize_map = Op::GetAttr<FRTQuantize>("FRTQuantize");
  // const auto& bounds = src.GetAttr<std::vector<int>>("bounds");
  // int num_samples = src.GetAttr<int>("num_samples");
  // TCalibInfo calib_info = CalibStatisticalAnalyse(idx, num_samples, bounds);
  int mode = src.GetAttr<int>("mode");
  int acc_dtype = src.GetAttr<int>("acc_dtype");
  int debug = src.GetAttr<int>("debug");
  TCalibInfo calib_info;
  TQuantizeConfig config{static_cast<TQuantizeConfig::Mode>(mode), acc_dtype};

  std::unordered_map<Node*, NodeEntry> quantized_var;
  std::unordered_map<Node*, int> reverse_mirror;
  TScaleMap scale_map;
  scale_map.resize(idx.num_node_entries());
  std::vector<NodeEntry> debug_outputs;

  auto transform = [&](uint32_t nid, const NodePtr& n, std::vector<NodeEntry>* ret) {
      std::cout << std::endl;
    LOG(INFO) << "transform: " << n->attrs.name;
    if (n->is_variable()) return false;
    if (quantize_map.count(n->op())) {
      NodePtr temp = MakeNode(n->op()->name.c_str(), n->attrs.name, n->inputs, n->attrs.dict).node;
      for (size_t i = 0; i < temp->inputs.size(); ++i) {
        const auto& e = temp->inputs[i];
        if (e.node->is_variable()) {
          if (quantized_var.count(e.node.get())) {
            n->inputs[i] = quantized_var.at(e.node.get());
          } else {
            NodeEntry scale = MakeNode("scale", e.node->attrs.name + "_scale",
              {e}, {{"mode", mode == 0 ? "real" : "base2"}});
            scale_map[idx.entry_id(e)] = scale;

            NodeEntry quantize = MakeQuantizeNode(e, scale);
            quantized_var.emplace(e.node.get(), quantize);
            temp->inputs[i] = quantize;
          }
        }
      }

      uint32_t num_out = n->num_outputs();
      CHECK_EQ(num_out, 1);
      /// assume only one output;

      auto fquantize = quantize_map[n->op()];
      std::vector<NodeEntry> qoutputs = fquantize(nid, temp, idx, calib_info, scale_map, config);
      reverse_mirror.emplace(qoutputs[0].node.get(), nid);
      // update scale map
      scale_map[idx.entry_id(nid, 0)] = qoutputs[1];

      std::vector<NodeEntry> outputs;
      outputs.reserve(num_out);
      outputs.emplace_back(qoutputs[0]);
      if (debug) {
        debug_outputs.emplace_back(qoutputs[0]);
      }
      *ret = std::move(outputs);
      return true;
    } else {
      LOG(FATAL) << n->op()->name << " cannot be quantized yet.";
      return false;
    }
  };

  Graph ret = compiler::GraphTransform(src, transform);
  LOG(INFO) << "prepare outputs";
  const std::vector<NodeEntry>& src_outputs = debug ? debug_outputs : ret.outputs;
  // const std::vector<NodeEntry>& src_outputs = ret.outputs;
  std::vector<NodeEntry> outputs;
  outputs.reserve(src_outputs.size());
  for (const auto& e : src_outputs) {
    uint32_t old_nid = reverse_mirror.at(e.node.get());
    NodeEntry scale = scale_map[idx.entry_id(old_nid, e.index)];
    NodeEntry dequantize = MakeDequantizeNode(e, scale);
    outputs.emplace_back(dequantize);
  }
  ret.outputs = std::move(outputs);
  LOG(INFO) << "quantize exit";
  return ret;
}


Graph QuantizeGraph(nnvm::Graph&& src) {
  return QuantizeGraphWithoutCalibration(std::move(src));
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
