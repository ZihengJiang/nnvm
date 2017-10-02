/*!
 *  Copyright (c) 2017 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/top/nn.h>
#include <nnvm/top/tensor.h>
#include "./nn_common.h"
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {

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
    .set_default(kInt16)
    .add_enum("int8", kInt8)
    .add_enum("int16", kInt16);
  }
  // constants
  static const constexpr int kData = 0;
  static const constexpr int kWeight = 1;
  static const constexpr int kBias = 2;
};


// dense
DMLC_REGISTER_PARAMETER(QuantizedDenseParam);

inline bool QuantizedDenseInferShape(const nnvm::NodeAttrs& attrs,
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

inline bool QuantizedDenseInferType(const nnvm::NodeAttrs& attrs,
                                    std::vector<int>* in_type,
                                    std::vector<int>* out_type) {
  const QuantizedDenseParam& param = nnvm::get<QuantizedDenseParam>(attrs.parsed);
  if (param.use_bias) {
    CHECK_EQ(in_type->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_type->size(), 2U) << "Input:[data, weight]";
  }
  CHECK_EQ(out_type->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_type, 0, param.out_type);
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
.set_attr<FInferShape>("FInferShape", QuantizedDenseInferShape)
.set_attr<FInferType>("FInferType", QuantizedDenseInferType)
.set_support_level(1);

NNVM_REGISTER_OP(dense)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    NodePtr qnode = Node::Create();
    qnode->attrs.op = nnvm::Op::Get("quantized_dense");
    qnode->attrs.name = "quantized_" + n->attrs.name;
    qnode->attrs.dict = n->attrs.dict;
    if (qnode->attrs.op->attr_parser) {
      qnode->attrs.op->attr_parser(&(qnode->attrs));
    }
    return qnode;
  });
}  // namespace top
}  // namespace nnvm
