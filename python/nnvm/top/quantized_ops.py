# pylint: disable=invalid-name, unused-argument
"""Quantization operators"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern

@reg.register_compute("quantize")
def compute_quantize(attrs, inputs, _):
    k = attrs.get_int('k')
    out_dtype = attrs['out_type']
    assert out_dtype == 'int8'
    data = inputs[0]
    scale = float(pow(2, 7) - 1) / pow(2, k)
    mul = tvm.compute(data.shape, lambda *i: data(*i) * scale)
    cast = tvm.compute(data.shape, lambda *i: tvm.select(mul(*i) < 0, (mul(*i) - 0.5).astype(out_dtype), (mul(*i) + 0.5).astype(out_dtype)))
    return cast


@reg.register_schedule("quantize")
def schedule_quantize(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("dequantize")
def compute_dequantize(attrs, inputs, _):
    k = attrs.get_int('k')
    out_dtype = attrs['out_type']
    assert out_dtype == 'int8'
    data = inputs[0]
    scale = pow(2, k) / float(pow(2, 7) - 1)
    mul = tvm.compute(data.shape, lambda *i: data(*i) * scale)
    return mul


@reg.register_schedule("dequantize")
def schedule_dequantize(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])

@reg.register_compute("quantized_dense")
def compute_quantized_dense(attrs, inputs, _):
    shift_out = attrs.get_int('shift_out')
    out_dtype = attrs['out_type']
    cmp_dtype = 'int32' # compute data type
    data   = inputs[0]
    weight = inputs[1]
    m, l = data.shape
    n, _ = weight.shape

    k = tvm.reduce_axis((0, l), name='k')
    out_i16 = tvm.compute((m, n),
        lambda i, j: tvm.sum(data[i][k].astype(cmp_dtype) * weight[j][k].astype(cmp_dtype), axis=k))

    if attrs.get_bool("use_bias"):
        bias = inputs[2]
        shift_bias = attrs.get_int("shift_bias")
        bias_i16 = topi.right_shift(topi.cast(bias, cmp_dtype), shift_bias)
        out_i16 = topi.broadcast_add(out_i16, bias_i16)

    if out_dtype == 'int8':
        out_i16 = topi.right_shift(out_i16, shift_out)
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
    cmp_dtype = 'int32' # compute data type

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
