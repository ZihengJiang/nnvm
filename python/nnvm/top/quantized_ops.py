# pylint: disable=invalid-name, unused-argument
"""Quantization operators"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern


@reg.register_compute("quantized_elemwise_add")
def compute_quantized_elemwise_add(attrs, inputs, _):
    a_shift = attrs.get_int('a_shift')
    b_shift = attrs.get_int('b_shift')
    c_shift = attrs.get_int('c_shift')
    out_dtype = attrs['out_type']
    cmp_dtype = 'int16'

    a_i8 = inputs[0]
    b_i8 = inputs[1]
    a_i16 = topi.right_shift(topi.cast(a_i8, cmp_dtype), a_shift)
    b_i16 = topi.right_shift(topi.cast(a_i8, cmp_dtype), a_shift)
    c_i16 = tvm.compute(a_i16.shape, lambda *i: a_i16(*i) + b_i16(*i))
    c_i16 = topi.right_shift(c_i16, c_shift)
    if out_dtype == 'int8':
        c_i8 = topi.cast(topi.clip(c_i16, -127, 127), out_dtype)
        return c_i8
    else:
        return c_i16

@reg.register_schedule("quantized_elemwise_add")
def schedule_quantized_elemwise_add(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_broadcast_add")
def compute_quantized_broadcast_add(attrs, inputs, _):
    pass

@reg.register_schedule("quantized_broadcast_add")
def schedule_quantized_broadcast_add(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_broadcast_mul")
def compute_quantized_broadcast_mul(attrs, inputs, _):
    pass

@reg.register_schedule("quantized_broadcast_mul")
def schedule_quantized_broadcast_mul(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_dense")
def compute_quantized_dense(attrs, inputs, _):
    shift = attrs.get_int('shift')
    out_dtype = attrs['out_type']
    cmp_dtype = 'int32' # compute data type
    data   = inputs[0]
    weight = inputs[1]
    m, l = data.shape
    n, _ = weight.shape

    k = tvm.reduce_axis((0, l), name='k')
    out_i16 = tvm.compute((m, n),
        lambda i, j: tvm.sum(data[i][k].astype(cmp_dtype) * weight[j][k].astype(cmp_dtype), axis=k))
    out_i16 = topi.right_shift(out_i16, shift)

    if out_dtype == 'int8':
        return topi.cast(topi.clip(out_i16, -127, 127), out_dtype)
    else:
        return out_i16

@reg.register_schedule("quantized_dense")
def schedule_quantized_dense(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_conv2d")
def compute_quantized_conv2d(attrs, inputs, _):
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    channels = attrs.get_int("channels")
    layout = attrs["layout"]
    shift = attrs.get_int('shift')
    out_dtype = attrs['out_type']
    cmp_dtype = 'int16' # compute data type

    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support group==1 conv2d"

    data = inputs[0]
    kernel = inputs[1]

    N, CI, HI, WI = [x.value for x in data.shape]
    assert N == 1, "quantized_conv2d only support N == 1 for now"
    CO, _, HK, WK = [x.value for x in kernel.shape]
    HO = (HI + 2*padding[0] - HK) / strides[0] + 1
    WO = (WI + 2*padding[1] - WK) / strides[1] + 1

    data_pad = topi.nn.pad(data, [0, 0, padding[0], padding[1]])

    rc = tvm.reduce_axis((0, CI), name='rc')
    rh = tvm.reduce_axis((0, HK), name='rh')
    rw = tvm.reduce_axis((0, WK), name='rw')

    out_i16 = tvm.compute((N, CO, HO, WO), lambda n, c, h, w:
        tvm.sum(data_pad[n, rc, h*strides[0] + rh, w*strides[1] + rw].astype(cmp_dtype) *
                kernel[c, rc, rh, rw].astype(cmp_dtype), axis=[rc, rh, rw]))

    if out_dtype == 'int8':
        return topi.cast(topi.clip(topi.right_shift(out_i16, shift), -127, 127), out_dtype)
    else:
        return out_i16

@reg.register_schedule("quantized_conv2d")
def schedule_quantized_conv2d(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_avg_pool2d")
def compute_quantized_avg_pool2d(attrs, inputs, _):
    pass

@reg.register_schedule("quantized_avg_pool2d")
def schedule_quantized_avg_pool2d(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])
