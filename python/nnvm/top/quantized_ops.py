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
    scale = float(pow(2, 7) - 0.5) / pow(2, k)
    scaled_data = tvm.compute(data.shape, lambda *i: data(*i) * scale)
    cliped_data = topi.clip(scaled_data, -127, 127)
    cast = tvm.compute(cliped_data.shape, lambda *i: tvm.select(cliped_data(*i) < 0,
        (cliped_data(*i) - 0.5).astype(out_dtype), (cliped_data(*i) + 0.5).astype(out_dtype)), name='cast')
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
    scale = pow(2, k) / float(pow(2, 7) - 0.5)
    scaled_data = tvm.compute(data.shape, lambda *i: (data(*i)) * scale)
    return scaled_data

@reg.register_schedule("dequantize")
def schedule_dequantize(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_dense")
def compute_quantized_dense(attrs, inputs, _):
    shift = attrs.get_int('shift')
    out_dtype = attrs['out_type']
    cmp_dtype = 'int32' # compute data type
    assert attrs.get_bool("use_bias") == False

    data   = inputs[0]
    weight = inputs[1]
    m, l = data.shape
    n, _ = weight.shape

    k = tvm.reduce_axis((0, l), name='k')
    out = tvm.compute((m, n),
        lambda i, j: tvm.sum(data[i][k].astype(cmp_dtype) * weight[j][k].astype(cmp_dtype), axis=k))

    if out_dtype == 'int8':
        assert shift >= 1
        shift_out = topi.noise_rshift(out, shift)
        return topi.cast(topi.clip(shift_out, -127, 127), out_dtype)
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
    assert attrs.get_bool("use_bias") == False
    out = topi.nn.conv2d(inputs[0], inputs[1], strides, padding, out_dtype=cmp_dtype)

    if out_dtype == 'int8':
        assert shift >= 1
        shift_out = topi.noise_rshift(out, shift)
        return topi.cast(topi.clip(shift_out, -127, 127), out_dtype)
    else:
        return out

@reg.register_schedule("quantized_conv2d")
def schedule_quantized_conv2d(_, outs, target):
    with tvm.target.create(target):
        return topi.generic.schedule_conv2d_nchw(outs)
