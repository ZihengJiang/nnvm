/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantized_ops.cc
 * \brief Quantization operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <nnvm/top/tensor.h>
#include <nnvm/compiler/op_attr_types.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "../broadcast_op_common.h"

namespace nnvm {
namespace top {

using compiler::FQuantize;
using compiler::FSeparateBias;

template<typename TParam>
inline bool QuantizeType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_type,
                            std::vector<int>* out_type) {
  const TParam& param = nnvm::get<TParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
  return true;
}

inline FQuantize DefaultQuantize(const char* op_name) {
  // identity, reshape, flatten, max_pool2d
  // dense, conv2d
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map) {
    NodeEntry qnode = MakeNode(op_name, n->attrs.name,
      n->inputs, n->attrs.dict);
    return qnode.node;
  };
}


inline FQuantize ExpandQuantize(const char* op_name) {
  // relu, global_avg_pool
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map) {
    std::string node_name = n->attrs.name;

    const auto& inputs = idx[nid].inputs;
    int k_i = scale_map[inputs[0].node_id];
    int k_o = scale_map[nid];
    int bit = (k_i - k_o);  // lshift
    if (bit < 0) {
      LOG(WARNING) << n->attrs.name << " increase the magnitude";
    }

    NodeEntry casti32 = MakeNode("cast", n->inputs[0].node->attrs.name + "_cast",
      {n->inputs[0]}, {{"dtype", "int32"}});
    NodeEntry qnode = MakeNode(op_name, node_name,
      {casti32}, n->attrs.dict);
    NodeEntry shift = qnode;
    if (bit > 0) {
      shift = MakeNode("noise_lshift", node_name + "_lshift",
        {shift}, {{"bit", std::to_string(bit)}});
    }
    NodeEntry clip = MakeNode("clip", node_name + "_clip",
      {shift}, {{"a_min", "-127"}, {"a_max", "127"}});
    NodeEntry casti8 = MakeNode("cast", node_name + "_cast",
      {clip}, {{"dtype", "int8"}});
    return casti8.node;
  };
}


inline FQuantize AdditionQuantize(const char* op_name) {
  // elemwise_add, broadcast_add
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map) {
    std::string node_name = n->attrs.name;

    const auto& inputs = idx[nid].inputs;
    int ka = scale_map[inputs[0].node_id];
    int kb = scale_map[inputs[1].node_id];
    int kc = scale_map[nid];

    int kmin = std::min(ka, kb);
    int sa = ka - kmin;
    int sb = kb - kmin;
    int sc = kc - kmin;

    NodeEntry lhs = n->inputs[0];
    NodeEntry rhs = n->inputs[1];
    std::string lname = lhs.node->attrs.name;
    std::string rname = rhs.node->attrs.name;
    if (sa >= 7) {
      LOG(INFO) << n->attrs.name << " skip add rhs: " << sa;
      // if (sa != sc) {
      //   LOG(WARNING) << "skip addition but change the magnitude";
      //   int bit = sc - sa;
      //   CHECK_GT(bit, 0);
      //   NodeEntry round = MakeNode("stochastic_round", lname + "_round",
      //     {lhs}, {{"bit", std::to_string(bit)}});
      //   return MakeNode("right_shift", lname + "_shift",
      //     {round}, {{"bit", std::to_string(bit)}}).node;
      // }
      return MakeNode("identity", lname + "_skip_add", {lhs}).node;
    }
    if (sb >= 7) {
      LOG(INFO) << n->attrs.name << " skip add lhs: " << sb;
      // if (sb != sc) {
      //   LOG(WARNING) << "skip addition but change the magnitude";
      //   int bit = sc - sb;
      //   CHECK_GT(bit, 0);
      //   NodeEntry round = MakeNode("stochastic_round", rname + "_round",
      //     {lhs}, {{"bit", std::to_string(bit)}});
      //   return MakeNode("right_shift", rname + "_shift",
      //     {round}, {{"bit", std::to_string(bit)}}).node;
      // }
      return MakeNode("identity", rname + "_skip_add", {rhs}).node;
    }

    CHECK(sa >= 0 && sb >= 0);
    NodeEntry lhs_cast = MakeNode("cast", lname + "_cast",
      {lhs}, {{"dtype", "int16"}});
    NodeEntry rhs_cast = MakeNode("cast", rname + "_cast",
      {rhs}, {{"dtype", "int16"}});
    NodeEntry lhs_shift = lhs_cast;
    if (sa != 0) {
      lhs_shift = MakeNode("noise_lshift", lname + "_lshift",
        {lhs_cast}, {{"bit", std::to_string(sa)}});
    }
    NodeEntry rhs_shift = rhs_cast;
    if (sb != 0) {
      rhs_shift = MakeNode("noise_lshift", rname + "_lshift",
        {rhs_cast}, {{"bit", std::to_string(sb)}});
    }
    NodeEntry qnode = MakeNode(op_name, node_name + "_quantized",
      {lhs_shift, rhs_shift});
    NodeEntry out_shift = qnode;
    if (sc != 0) {
      if (sc > 0) {
        out_shift = MakeNode("stochastic_round", node_name + "_round",
          {out_shift}, {{"bit", std::to_string(sc)}});
        out_shift = MakeNode("right_shift", node_name + "_rshift",
          {out_shift}, {{"bit", std::to_string(sc)}});
      } else {
        LOG(WARNING) << "elemwise add lower the precision: " << sc;
        out_shift = MakeNode("noise_lshift", node_name + "_lshift",
          {qnode}, {{"bit", std::to_string(-sc)}});
      }
    }
    NodeEntry clip = MakeNode("clip", node_name + "_clip",
      {out_shift}, {{"a_min", "-127"}, {"a_max", "127"}});
    NodeEntry cast = MakeNode("cast", node_name + "_cast",
      {clip}, {{"dtype", "int8"}});
    return cast.node;
  };
}


inline FQuantize MultiplicationQuantize(const char* op_name) {
  // elemwise_mul, broadcast_mul
  return [=] (uint32_t nid,
              const NodePtr& n,
              const IndexedGraph& idx,
              const std::vector<int>& scale_map) {
    const auto& inputs = idx[nid].inputs;
    int ka = scale_map[inputs[0].node_id];
    int kb = scale_map[inputs[1].node_id];
    int kc = scale_map[nid];
    // qc = f(qa, qb) = f(va * 2^7 / 2^ka, vb * 2^7 / 2^kb) = f(va, vb) * 2^(14 - a + b)
    // |vc| * 2^(14 - a - b) < 2^kc * 2^(14 - ka - kb)
    // qc ~ (14 + kc - ka - kb)
    int bit = kc + 7 - (ka + kb);
    // CHECK_GT(bit, 0);
    // CHECK_LT(bit, 15);

    std::string node_name = n->attrs.name;
    NodeEntry lhs = MakeNode("cast", n->inputs[0].node->attrs.name + "_cast",
      {n->inputs[0]}, {{"dtype", "int32"}});
    NodeEntry rhs = MakeNode("cast", n->inputs[1].node->attrs.name + "_cast",
      {n->inputs[1]}, {{"dtype", "int32"}});
    NodeEntry qnode = MakeNode(op_name, node_name + "_quantized",
      {lhs, rhs}, n->attrs.dict);

    NodeEntry shift = qnode;
    if (bit > 0) {
      NodeEntry round = MakeNode("stochastic_round", node_name + "_round",
        {qnode}, {{"bit", std::to_string(bit)}});
      shift = MakeNode("right_shift", node_name + "_rshift",
        {round}, {{"bit", std::to_string(bit)}});
    }

    NodeEntry clip = MakeNode("clip", node_name + "_clip",
      {shift}, {{"a_min", "-127"}, {"a_max", "127"}});
    NodeEntry cast = MakeNode("cast", node_name + "_cast",
      {clip}, {{"dtype", "int8"}});
    return cast.node;
  };
}


// quantize

struct QuantizeParam : public dmlc::Parameter<QuantizeParam> {
  int k;
  int out_type;

  DMLC_DECLARE_PARAMETER(QuantizeParam) {
    DMLC_DECLARE_FIELD(k);
    DMLC_DECLARE_FIELD(out_type)
    .set_default(kInt8)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16)
    .add_enum("int32", kInt32);
  };
};

DMLC_REGISTER_PARAMETER(QuantizeParam);

NNVM_REGISTER_OP(quantize)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", QuantizeType<QuantizeParam>)
.add_argument("data", "Tensor", "The input tensor.")
.add_arguments(QuantizeParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizeParam>);


// dequantize
inline bool DequantizeType(const nnvm::NodeAttrs& attrs,
                           std::vector<int>* in_type,
                           std::vector<int>* out_type) {
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, 0);
  return true;
}

NNVM_REGISTER_OP(dequantize)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<FInferType>("FInferType", DequantizeType)
.add_argument("data", "Tensor", "The input tensor.")
.add_arguments(QuantizeParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizeParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizeParam>);


// stochastic round

struct StochasticRoundParam : public dmlc::Parameter<StochasticRoundParam> {
  int bit;
  DMLC_DECLARE_PARAMETER(StochasticRoundParam) {
    DMLC_DECLARE_FIELD(bit);
  };
};

DMLC_REGISTER_PARAMETER(StochasticRoundParam);

NNVM_REGISTER_ELEMWISE_UNARY_OP(stochastic_round)
.add_arguments(StochasticRoundParam::__FIELDS__())
.set_attr_parser(ParamParser<StochasticRoundParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<StochasticRoundParam>);


// noise_shift

struct NoiseShiftParam : public dmlc::Parameter<NoiseShiftParam> {
  int bit;
  DMLC_DECLARE_PARAMETER(NoiseShiftParam) {
    DMLC_DECLARE_FIELD(bit);
  };
};

DMLC_REGISTER_PARAMETER(NoiseShiftParam);

NNVM_REGISTER_ELEMWISE_UNARY_OP(noise_lshift)
.add_arguments(NoiseShiftParam::__FIELDS__())
.set_attr_parser(ParamParser<NoiseShiftParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<NoiseShiftParam>);


// quantized elemwise_add

NNVM_REGISTER_OP(elemwise_add)
.set_attr<FQuantize>("FQuantize", AdditionQuantize("elemwise_add"));


// quantized broadcast_add

NNVM_REGISTER_OP(broadcast_add)
.set_attr<FQuantize>("FQuantize", AdditionQuantize("broadcast_add"));


// quantized elemwise_mul

NNVM_REGISTER_OP(elemwise_mul)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("elemwise_mul"));


// quantized broadcast_mul

NNVM_REGISTER_OP(broadcast_mul)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("broadcast_mul"));


// quantized identity

NNVM_REGISTER_OP(identity)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("identity"));


// quantized reshape

NNVM_REGISTER_OP(reshape)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("reshape"));


// quantized flatten

NNVM_REGISTER_OP(flatten)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("flatten"));


// quantized relu

NNVM_REGISTER_OP(relu)
.set_attr<FQuantize>("FQuantize", ExpandQuantize("relu"));


// quantized dense

struct QuantizedDenseParam : public dmlc::Parameter<QuantizedDenseParam> {
  int units;
  bool use_bias;
  int shift;
  int out_type;

  DMLC_DECLARE_PARAMETER(QuantizedDenseParam) {
    DMLC_DECLARE_FIELD(units).set_lower_bound(1)
    .describe("Number of hidden units of the dense transformation.");
    DMLC_DECLARE_FIELD(use_bias).set_default(true)
    .describe("Whether to use bias parameter");
    DMLC_DECLARE_FIELD(shift)
    .set_default(0);
    DMLC_DECLARE_FIELD(out_type)
    .set_default(kInt32)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16)
    .add_enum("int32", kInt32);
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};

DMLC_REGISTER_PARAMETER(QuantizedDenseParam);

inline bool QuantizedDenseShape(const nnvm::NodeAttrs& attrs,
                                std::vector<TShape>* in_shape,
                                std::vector<TShape>* out_shape) {
  const QuantizedDenseParam& param = nnvm::get<QuantizedDenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);
  // reverse infer
  if ((*out_shape)[0].ndim() != 0) {
    TShape dshape = (*out_shape)[0];
    dshape[dshape.ndim() - 1] = 0;
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedDenseParam::kData, dshape);
  }
  dim_t num_inputs = 0;
  if ((*in_shape)[QuantizedDenseParam::kData].ndim() != 0) {
    TShape oshape = (*in_shape)[QuantizedDenseParam::kData];
    num_inputs = oshape[oshape.ndim() - 1];
    oshape[oshape.ndim() - 1] = param.units;
    NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0, oshape);
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedDenseParam::kWeight,
                          TShape({param.units, num_inputs}));
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedDenseParam::kBias, TShape({param.units}));
  }
  return true;
}

NNVM_REGISTER_OP(quantized_dense)
.add_argument("data", "nD Tensor", "Input data.")
.add_argument("weight", "2D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(QuantizedDenseParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizedDenseParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizedDenseParam>)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<QuantizedDenseParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<QuantizedDenseParam>)
.set_attr<FInferShape>("FInferShape", QuantizedDenseShape)
.set_attr<FInferType>("FInferType", QuantizeType<QuantizedDenseParam>)
.set_support_level(1);

NNVM_REGISTER_OP(dense)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("quantized_dense"))
.set_attr<FSeparateBias>("FSeparateBias", [] (const NodePtr& n) {
  const DenseParam& param = nnvm::get<DenseParam>(n->attrs.parsed);
  if (param.use_bias == false) return std::vector<NodeEntry>({NodeEntry{n, 0, 0}});
  std::unordered_map<std::string, std::string> dict = n->attrs.dict;
  dict["use_bias"] = "False";
  NodeEntry node = MakeNode(n->op()->name.c_str(), n->attrs.name,
    {n->inputs[0], n->inputs[1]}, dict);
  NodeEntry node_with_bias = MakeNode("broadcast_add", n->attrs.name + "_add_bias",
    {node, n->inputs[2]});
  return std::vector<NodeEntry>({node_with_bias});
});


// quantized conv2d

struct QuantizedConv2DParam : public dmlc::Parameter<QuantizedConv2DParam> {
  int channels;
  TShape kernel_size;
  TShape strides;
  TShape padding;
  TShape dilation;
  int groups;
  int layout;
  bool use_bias;
  int shift;
  int out_type;

  DMLC_DECLARE_PARAMETER(QuantizedConv2DParam) {
    DMLC_DECLARE_FIELD(channels)
      .describe("The dimensionality of the output space"
                "i.e. the number of output channels in the convolution.");
    DMLC_DECLARE_FIELD(kernel_size)
      .describe("Specifies the dimensions of the convolution window.");
    DMLC_DECLARE_FIELD(strides).set_default(TShape({1, 1}))
      .describe("Specifies the strides of the convolution.");
    DMLC_DECLARE_FIELD(padding).set_default(TShape({0, 0}))
      .describe("If padding is non-zero, then the input is implicitly zero-padded"
                "on both sides for padding number of points");
    DMLC_DECLARE_FIELD(dilation).set_default(TShape({1, 1}))
      .describe("Specifies the dilation rate to use for dilated convolution.");
    DMLC_DECLARE_FIELD(groups).set_default(1)
      .describe("Controls the connections between inputs and outputs."
                "At groups=1, all inputs are convolved to all outputs."
                "At groups=2, the operation becomes equivalent to having two convolution"
                "layers side by side, each seeing half the input channels, and producing"
                "half the output channels, and both subsequently concatenated.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCHW", kNCHW)
      .add_enum("NHWC", kNHWC)
      .set_default(kNCHW)
      .describe("Dimension ordering of data and weight. Can be 'NCHW', 'NHWC', etc."
                "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
                "dimensions respectively. Convolution is applied on the 'H' and"
                "'W' dimensions.");
    DMLC_DECLARE_FIELD(use_bias).set_default(true)
      .describe("Whether the layer uses a bias vector.");
    DMLC_DECLARE_FIELD(shift)
    .set_default(0);
    DMLC_DECLARE_FIELD(out_type)
    .set_default(kInt32)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16)
    .add_enum("int32", kInt32);
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};


DMLC_REGISTER_PARAMETER(QuantizedConv2DParam);

inline bool QuantizedConv2DShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape>* in_shape,
                                 std::vector<TShape>* out_shape) {
  const QuantizedConv2DParam& param = nnvm::get<QuantizedConv2DParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_shape->size(), 1U);

  TShape dshape = in_shape->at(0);
  if (dshape.ndim() == 0) return false;
  dshape = ConvertLayout(dshape, param.layout, kNCHW);

  CHECK_EQ(dshape.ndim(), 4U) << "Input data should be 4D";
  CHECK_EQ(param.kernel_size.ndim(), 2U);
  CHECK_EQ(param.strides.ndim(), 2U)
      << "incorrect stride size: " << param.strides;
  CHECK_EQ(param.dilation.ndim(), 2U)
      << "incorrect dilate size: " << param.dilation;
  CHECK_EQ(dshape[1] % param.groups, 0U)
      << "input channels must divide group size";
  CHECK_EQ(param.channels % param.groups, 0U)
      << "output channels must divide group size";

  TShape wshape({param.channels / param.groups,
                 dshape[1] / param.groups,
                 param.kernel_size[0],
                 param.kernel_size[1]});

  wshape = ConvertLayout(wshape, kNCHW, param.layout);
  wshape[0] *= param.groups;

  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedConv2DParam::kWeight, wshape);
  if (param.use_bias) {
    NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape,
                            QuantizedConv2DParam::kBias, TShape({param.channels}));
  }
  // dilation
  dim_t dilated_ksize_y = 1 + (param.kernel_size[0] - 1) * param.dilation[0];
  dim_t dilated_ksize_x = 1 + (param.kernel_size[1] - 1) * param.dilation[1];
  TShape oshape({dshape[0], param.channels, 0, 0});
  if (dshape[2] != 0) {
    oshape[2] = (dshape[2] + param.padding[0] * 2 - dilated_ksize_y) / param.strides[0] + 1;
  }
  if (dshape[3] != 0) {
    oshape[3] = (dshape[3] + param.padding[1] * 2 - dilated_ksize_x) / param.strides[1] + 1;
  }
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_shape, 0,
                           ConvertLayout(oshape, kNCHW, param.layout));
  // Perform incomplete shape inference. Fill in the missing values in data shape.
  // 1) We can always fill in the batch_size.
  // 2) We can back-calculate the input height/width if the corresponding stride is 1.
  oshape = ConvertLayout((*out_shape)[0], param.layout, kNCHW);
  dshape[0] = oshape[0];
  if (oshape[2] && param.strides[0] == 1) {
    dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param.padding[0];
  }
  if (oshape[3] && param.strides[1] == 1) {
    dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param.padding[1];
  }
  NNVM_ASSIGN_INPUT_SHAPE(attrs, *in_shape, QuantizedConv2DParam::kData,
                          ConvertLayout(dshape, kNCHW, param.layout));
  // Check whether the kernel sizes are valid
  if (dshape[2] != 0) {
    CHECK_LE(dilated_ksize_y, dshape[2] + 2 * param.padding[0])
      << "kernel size exceed input";
  }
  if (dshape[3] != 0) {
    CHECK_LE(dilated_ksize_x, dshape[3] + 2 * param.padding[1])
        << "kernel size exceed input";
  }
  return true;
}

inline bool QuantizedConv2DType(const nnvm::NodeAttrs& attrs,
                                std::vector<int>* in_type,
                                std::vector<int>* out_type) {
  const QuantizedConv2DParam& param = nnvm::get<QuantizedConv2DParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_type->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_type->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
  return true;
}

NNVM_REGISTER_OP(quantized_conv2d)
.add_argument("data", "4D Tensor", "Input data.")
.add_argument("weight", "4D Tensor", "Weight matrix.")
.add_argument("bias", "1D Tensor", "Bias parameter.")
.add_arguments(QuantizedConv2DParam::__FIELDS__())
.set_attr_parser(ParamParser<QuantizedConv2DParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<QuantizedConv2DParam>)
.set_attr<FListInputNames>("FListInputNames", UseBiasListInputNames<QuantizedConv2DParam>)
.set_attr<FInferShape>("FInferShape", QuantizedConv2DShape)
.set_attr<FInferType>("FInferType", QuantizedConv2DType)
.set_num_outputs(1)
.set_num_inputs(UseBiasNumInputs<QuantizedConv2DParam>)
.set_support_level(2);

NNVM_REGISTER_OP(conv2d)
.set_attr<FQuantize>("FQuantize", MultiplicationQuantize("quantized_conv2d"))
.set_attr<FSeparateBias>("FSeparateBias", [] (const NodePtr& n) {
  const Conv2DParam& param = nnvm::get<Conv2DParam>(n->attrs.parsed);
  if (param.use_bias == false) return std::vector<NodeEntry>({NodeEntry{n, 0, 0}});
  std::unordered_map<std::string, std::string> dict = n->attrs.dict;
  dict["use_bias"] = "False";
  NodeEntry node = MakeNode(n->op()->name.c_str(), n->attrs.name,
    {n->inputs[0], n->inputs[1]}, dict);
  NodeEntry bias = n->inputs[2];
  NodeEntry expand = MakeNode("expand_dims", bias.node->attrs.name + "_expand",
    {bias}, {{"axis", "1"}, {"num_newaxis", "2"}});
  NodeEntry node_with_bias = MakeNode("broadcast_add", n->attrs.name + "_add_bias",
    {node, expand});
  return std::vector<NodeEntry>({node_with_bias});
});


// quantized max_pool2d

NNVM_REGISTER_OP(max_pool2d)
.set_attr<FQuantize>("FQuantize", DefaultQuantize("max_pool2d"));


// quantized global_avg_pool2d

NNVM_REGISTER_OP(global_avg_pool2d)
.set_attr<FQuantize>("FQuantize", ExpandQuantize("global_avg_pool2d"));

}  // namespace top
}  // namespace nnvm
