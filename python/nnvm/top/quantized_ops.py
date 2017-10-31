# pylint: disable=invalid-name, unused-argument
"""Quantization operators"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int
from . import registry as reg
from .registry import OpPattern
import numpy as np

@tvm.register_func("stochastic_round")
def stochastic_round(in_arr, out_arr, bit):
    dtype = in_arr.dtype
    iarr = in_arr.asnumpy()

    sign = np.sign(iarr)
    iarr = np.abs(iarr)
    shape = in_arr.shape
    threshold = np.random.randint(0, pow(2, bit), size=shape)
    low_kbit = np.bitwise_and(iarr, pow(2, bit) - 1)
    cond = (low_kbit > threshold)

    farr = np.bitwise_and(iarr, ~(pow(2, bit) - 1))
    limit = np.iinfo(dtype).max
    tarr = np.clip(farr.astype('int32') + pow(2, bit), -limit, limit).astype(dtype)
    tarr = np.bitwise_and(tarr, ~(pow(2, bit) - 1))
    oarr = np.where(cond, tarr, farr) * sign
    # central to zero
    # oarr = oarr * (farr != 0)
    out_arr.copyfrom(oarr.astype(dtype))
    # idx = ((31, ), (29, ))
    # print('iarr: {}'.format(iarr[idx]))
    # print('farr: {}'.format(farr[idx]))
    # print('tarr: {}'.format(tarr[idx]))
    # print('oarr: {}'.format(oarr[idx]))

@reg.register_compute("stochastic_round")
def compute_stochastic_round(attrs, inputs, _):
    bit = attrs.get_int('bit')
    assert bit > 0
    data = inputs[0]
    return tvm.extern(data.shape, [data],
        lambda ins, outs: tvm.intrin.call_packed("stochastic_round", ins[0], outs[0], bit),
        name='stochastic_round')

@reg.register_schedule("stochastic_round")
def schedule_stochastic_round(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])

def noise_rshift(data, bit):
    assert bit > 0
    rnd = tvm.extern(data.shape, [data],
        lambda ins, outs: tvm.intrin.call_packed("stochastic_round", ins[0], outs[0], bit),
        name='stochastic_round')
    return topi.right_shift(rnd, bit)


@tvm.register_func("noise_lshift")
def noise_lshift(in_arr, out_arr, bit):
    # print("noise lshift")
    # print("bit: {}".format(bit))
    dtype = in_arr.dtype
    iarr = in_arr.asnumpy()
    sign = np.sign(iarr)
    iarr = np.abs(iarr)
    shift_arr = np.left_shift(iarr, bit)

    value = pow(2, bit-1)-1
    noise = np.random.randint(-value, value+1)
    noise_arr = shift_arr + noise * (shift_arr != 0)
    oarr = noise_arr * sign
    out_arr.copyfrom(oarr)


@reg.register_compute("noise_lshift")
def compute_noise_lshift(attrs, inputs, _):
    bit = attrs.get_int('bit')
    assert bit > 0
    data = inputs[0]
    return tvm.extern(data.shape, [data],
        lambda ins, outs: tvm.intrin.call_packed("noise_lshift", ins[0], outs[0], bit),
        name='noise_lshift')

@reg.register_schedule("noise_lshift")
def schedule_noise_lshift(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@tvm.register_func("compute_scale")
def compute_scale(in_arr, out_arr):
    iarr = in_arr.asnumpy()
    arr = np.abs(iarr)
    scale = np.amax(arr)
    out_arr.copyfrom(scale.astype('float32'))

@reg.register_compute("scale_to_range")
def compute_scale_to_range(attrs, inputs, _):
    target_scale = attrs.get_float('scale')
    print("scale: {}".format(target_scale))
    data = inputs[0]
    real_scale = tvm.extern((1, ), [data],
        lambda ins, outs: tvm.intrin.call_packed("compute_scale", ins[0], outs[0]),
        name='compute_scale')
    scaled_data = tvm.compute(data.shape, lambda *i: data(*i) / real_scale[0] * target_scale, name='scale')
    print('dtype: {}'.format(scaled_data.dtype))
    return scaled_data


@reg.register_schedule("scale_to_range")
def schedule_scale_to_range(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


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
        shift_out = noise_rshift(out, shift)
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
    assert attrs.get_bool("use_bias") == False
    if groups == 1:
        out = topi.nn.conv2d(inputs[0], inputs[1], strides, padding, out_dtype=cmp_dtype)
    elif groups == get_const_int(inputs[0].shape[1]) and groups == channels:
        out = topi.nn.depthwise_conv2d_nchw(inputs[0], inputs[1], strides, padding, out_dtype=cmp_dtype)
    else:
        raise ValueError("not support arbitrary group number for now")

    if out_dtype == 'int8':
        assert shift >= 1
        shift_out = noise_rshift(out, shift)
        return topi.cast(topi.clip(shift_out, -127, 127), out_dtype)
    else:
        return out

@reg.register_schedule("quantized_conv2d")
def schedule_quantized_conv2d(_, outs, target):
    with tvm.target.create(target):
        return topi.generic.schedule_conv2d_nchw(outs)
