# pylint: disable=invalid-name, unused-argument
"""Quantization ops"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern

@reg.register_compute("quantized_dense")
def compute_quantized_dense(attrs, inputs, _):
    dtype = attrs['out_type']
    data   = inputs[0]
    weight = inputs[1]
    m, l = data.shape
    n, _ = weight.shape
    k = tvm.reduce_axis((0, l), name='k')
    return tvm.compute((m, n),
        lambda i, j: tvm.sum(data[i][k].astype(dtype) * weight[j][k].astype(dtype), axis=k))

@reg.register_schedule("quantized_dense")
def schedule_quantized_dense(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])
