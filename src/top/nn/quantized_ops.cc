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

using compiler::FQuantizedOp;
using compiler::FCalibrate;

template<typename TParam>
inline bool QuantizedOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_type,
                            std::vector<int>* out_type) {
  const TParam& param = nnvm::get<TParam>(attrs.parsed);
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
  return true;
}

inline FQuantizedOp MakeFQuantizedOp(const char* op_name) {
  return [=] (const NodePtr& n,
              const std::unordered_map<std::string, std::string>& dict) {
    auto ndict = n->attrs.dict;
    for (const auto& kv : dict) {
      ndict[kv.first] = kv.second;
    }
    NodeEntry qnode = MakeNode(op_name, "quantized_" + n->attrs.name,
      n->inputs, ndict);
    return qnode.node;
  };
}

inline FQuantizedOp MakeFQuantizedOpShiftUnaryInput(const char* op_name) {
  return [=] (const NodePtr& n,
              const std::unordered_map<std::string, std::string>& dict) {
    std::string node_name = "quantized_" + n->attrs.name;
    NodeEntry in = MakeNode("right_shift", node_name + "_shift",
      {n->inputs[0]}, {{"bit", dict.at("shift")}});
    NodeEntry qnode = MakeNode(op_name, node_name, {in});
    return qnode.node;
  };
}


inline FQuantizedOp MakeFQuantizedOpShiftBinaryInput(const char* op_name) {
  return [=] (const NodePtr& n,
              const std::unordered_map<std::string, std::string>& dict) {
    std::string node_name = "quantized_" + n->attrs.name;
    NodeEntry lhs = MakeNode("right_shift", node_name + "_lhs_shift",
      {n->inputs[0]}, {{"bit", dict.at("shift_a")}});
    NodeEntry rhs = MakeNode("right_shift", node_name + "_rhs_shift",
      {n->inputs[1]}, {{"bit", dict.at("shift_b")}});
    NodeEntry qnode = MakeNode("broadcast_add", node_name, {lhs, rhs});
    return qnode.node;
  };
}


// quantized elemwise_add

NNVM_REGISTER_OP(elemwise_add)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOpShiftBinaryInput("elemwise_add"))
.set_attr<FCalibrate>("FCalibrate",
  [](uint32_t nid, const nnvm::NodePtr& n, const IndexedGraph& idx,
     const std::vector<int>& base2_range,
     std::unordered_map<std::string, std::string>* dict) {
     const auto& inputs = idx[nid].inputs;
     int k_a = base2_range[inputs[0].node_id];
     int k_b = base2_range[inputs[1].node_id];
     int k_c = base2_range[nid];
     (*dict)["shift_a"] = std::to_string(k_a - k_c);
     (*dict)["shift_b"] = std::to_string(k_b - k_c);
  });


// quantized broadcast_add

NNVM_REGISTER_OP(broadcast_add)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOpShiftBinaryInput("broadcast_add"))
.set_attr<FCalibrate>("FCalibrate",
  [](uint32_t nid, const nnvm::NodePtr& n, const IndexedGraph& idx,
     const std::vector<int>& base2_range,
     std::unordered_map<std::string, std::string>* dict) {
     const auto& inputs = idx[nid].inputs;
     int k_a = base2_range[inputs[0].node_id];
     int k_b = base2_range[inputs[1].node_id];
     int k_c = base2_range[nid];
     (*dict)["shift_a"] = std::to_string(k_a - k_c);
     (*dict)["shift_b"] = std::to_string(k_b - k_c);
  });


// quantized broadcast_mul

NNVM_REGISTER_OP(broadcast_mul)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOpShiftBinaryInput("broadcast_mul"))
.set_attr<FCalibrate>("FCalibrate",
  [](uint32_t nid, const nnvm::NodePtr& n, const IndexedGraph& idx,
     const std::vector<int>& base2_range,
     std::unordered_map<std::string, std::string>* dict) {
     const auto& inputs = idx[nid].inputs;
     int k_a = base2_range[inputs[0].node_id];
     int k_b = base2_range[inputs[1].node_id];
     int k_c = base2_range[nid];

     // TODO(ziheng) add storage bits as argument
     int shift_a = k_a - 7;
     int shift_b = k_b - 7;
     int diff = (k_c - 7) - (shift_a + shift_b);
     if (diff > 0) {
        shift_a = diff / 2;
        shift_b = diff - (diff / 2);
     }

     (*dict)["shift_a"] = std::to_string(shift_a);
     (*dict)["shift_b"] = std::to_string(shift_b);
  });


// quantized reshape

NNVM_REGISTER_OP(reshape)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOp("reshape"));


// quantized relu

NNVM_REGISTER_OP(relu)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOp("relu"));


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
.describe(R"code(Applies a linear transformation: :math:`Y = XW^T + b`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **bias**: `(units,)`
- **out**: `(x1, x2, ..., xn, num_hidden)`

The learnable parameters include both ``weight`` and ``bias``.

If ``use_bias`` is set to be false, then the ``bias`` term is ignored.

)code" NNVM_ADD_FILELINE)
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
.set_attr<FInferType>("FInferType", QuantizedOpType<QuantizedDenseParam>)
.set_support_level(1);

NNVM_REGISTER_OP(dense)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOp("quantized_dense"))
.set_attr<FCalibrate>("FCalibrate",
  [](uint32_t nid, const nnvm::NodePtr& n, const IndexedGraph& idx,
     const std::vector<int>& base2_range,
     std::unordered_map<std::string, std::string>* dict) {
     const auto& inputs = idx[nid].inputs;
     int k_a = base2_range[inputs[0].node_id];
     int k_b = base2_range[inputs[1].node_id];
     int k_c = base2_range[nid];
     int shift = (k_c - (k_a + k_b) + 7);
     (*dict)["shift"] = std::to_string(shift);
     (*dict)["out_type"] = "int8";
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
.describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **bias**: (channels,)
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" NNVM_ADD_FILELINE)
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
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOp("quantized_conv2d"))
.set_attr<FCalibrate>("FCalibrate",
  [](uint32_t nid, const nnvm::NodePtr& n, const IndexedGraph& idx,
     const std::vector<int>& base2_range,
     std::unordered_map<std::string, std::string>* dict) {
     const auto& inputs = idx[nid].inputs;
     int k_a = base2_range[inputs[0].node_id];
     int k_b = base2_range[inputs[1].node_id];
     int k_c = base2_range[nid];
     int shift = (k_c - (k_a + k_b) + 7);
     (*dict)["shift"] = std::to_string(shift);
     (*dict)["out_type"] = "int8";
  });


// quantized max_pool2d

NNVM_REGISTER_OP(max_pool2d)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOp("max_pool2d"));


// quantized avg_global_pool2d

NNVM_REGISTER_OP(global_avg_pool2d)
.set_attr<FQuantizedOp>("FQuantizedOp", MakeFQuantizedOpShiftUnaryInput("global_avg_pool2d"))
.set_attr<FCalibrate>("FCalibrate",
  [](uint32_t nid, const nnvm::NodePtr& n, const IndexedGraph& idx,
     const std::vector<int>& base2_range,
     std::unordered_map<std::string, std::string>* dict) {
     const auto& inputs = idx[nid].inputs;
     int k_i = base2_range[inputs[0].node_id];
     int k_o = base2_range[nid];
     int shift = (k_i - k_o);
     (*dict)["shift"] = std::to_string(shift);
  });

}  // namespace top
}  // namespace nnvm
