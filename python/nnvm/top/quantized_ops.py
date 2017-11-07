# pylint: disable=invalid-name, unused-argument
"""Quantization operators"""
from __future__ import absolute_import

import tvm
import topi
from topi.util import get_const_int, get_const_tuple
from . import registry as reg
from .registry import OpPattern
import numpy as np

@tvm.register_func("debug")
def _debug(in_arr, out_arr):
    print('debug')
    print(in_arr.asnumpy())
    print('minimum: {}'.format(np.min(in_arr.asnumpy())))
    print('maximum: {}'.format(np.max(in_arr.asnumpy())))
    out_arr.copyfrom(in_arr)
    print('')

def debug(data):
    return tvm.extern(data.shape, [data],
        lambda ins, outs: tvm.intrin.call_packed("debug", ins[0], outs[0]),
        name='debug')


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
    oarr = oarr * (farr != 0)
    out_arr.copyfrom(oarr.astype(dtype))
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
#
# def noise_rshift(data, bit):
#     assert bit > 0
#     rnd = tvm.extern(data.shape, [data],
#         lambda ins, outs: tvm.intrin.call_packed("stochastic_round", ins[0], outs[0], bit),
#         name='stochastic_round')
#     return topi.right_shift(rnd, bit)
#
#
# @tvm.register_func("noise_lshift")
# def noise_lshift(in_arr, out_arr, bit):
#     # print("noise lshift")
#     # print("bit: {}".format(bit))
#     dtype = in_arr.dtype
#     iarr = in_arr.asnumpy()
#     sign = np.sign(iarr)
#     iarr = np.abs(iarr)
#     shift_arr = np.left_shift(iarr, bit)
#
#     value = pow(2, bit-1)-1
#     noise = np.random.randint(-value, value+1)
#     noise_arr = shift_arr + noise * (shift_arr != 0)
#     oarr = noise_arr * sign
#     out_arr.copyfrom(oarr)
#
#
# @reg.register_compute("noise_lshift")
# def compute_noise_lshift(attrs, inputs, _):
#     bit = attrs.get_int('bit')
#     assert bit > 0
#     data = inputs[0]
#     return tvm.extern(data.shape, [data],
#         lambda ins, outs: tvm.intrin.call_packed("noise_lshift", ins[0], outs[0], bit),
#         name='noise_lshift')
#
# @reg.register_schedule("noise_lshift")
# def schedule_noise_lshift(_, outs, target):
#     return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantize")
def compute_quantize(attrs, inputs, _):
    out_dtype = attrs['out_type']
    assert out_dtype == 'int8'
    data = inputs[0]
    scale = inputs[1]

    if scale.dtype == 'float32':
        scaled_data = tvm.compute(data.shape, lambda *i: data(*i) * (pow(2, 7) - 0.5) / scale[0])
    elif scale.dtype == 'int32':
        two = tvm.const(2.0, 'float32')
        scaled_data = tvm.compute(data.shape,
            lambda *i: data(*i) * (pow(2, 7) - 0.5) / tvm.power(two, scale[0].astype('float32')))
    else:
        raise ValueError

    cliped_data = topi.clip(scaled_data, -127, 127)
    cast = tvm.compute(cliped_data.shape, lambda *i: tvm.select(cliped_data(*i) < 0,
        (cliped_data(*i) - 0.5).astype(out_dtype), (cliped_data(*i) + 0.5).astype(out_dtype)), name='cast')
    return cast


@reg.register_schedule("quantize")
def schedule_quantize(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@tvm.register_func("add_noise")
def _add_noise(in_arr, out_arr, low, high):
    iarr = in_arr.asnumpy()
    assert iarr.dtype == 'float32'
    noise = np.random.uniform(0.0, 0.5-1e-6, iarr.shape) * (iarr != 0)
    iarr = iarr + noise
    out_arr.copyfrom(iarr)

def add_noise(data, low, high):
    return tvm.extern(data.shape, [data],
        lambda ins, outs: tvm.intrin.call_packed("add_noise", ins[0], outs[0], low, high),
        name='add_noise')

@reg.register_compute("dequantize")
def compute_dequantize(attrs, inputs, _):
    data = inputs[0]
    scale = inputs[1]

    if scale.dtype == 'float32':
        noise_data = data
        # noise_data = add_noise(topi.cast(data, 'float32'), -0.5, 0.5)
        scaled_data = tvm.compute(data.shape, lambda *i: (noise_data(*i)) * scale[0] / (pow(2, 7) - 0.5))
    elif scale.dtype == 'int32':
        two = tvm.const(2.0, 'float32')
        scaled_data = tvm.compute(data.shape,
            lambda *i: data(*i) * tvm.power(two, scale[0].astype('float32')) / (pow(2, 7) - 0.5))
    else:
        raise ValueError

    return scaled_data

@reg.register_schedule("dequantize")
def schedule_dequantize(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])


@reg.register_compute("quantized_dense")
def compute_quantized_dense(attrs, inputs, _):
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
        assert out_dtype == cmp_dtype
        return out

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
    out_dtype = attrs['out_type']
    cmp_dtype = 'int32' # compute data type

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
        assert out_dtype == cmp_dtype
        return out

@reg.register_schedule("quantized_conv2d")
def schedule_quantized_conv2d(_, outs, target):
    with tvm.target.create(target):
        return topi.generic.schedule_conv2d_nchw(outs)


@tvm.register_func("base2_round")
def _base2_round(in_arr, out_arr):
    precision = 32
    num = in_arr.asnumpy()[0]
    num = float(abs(num))

    # precision for extreme case
    if num < 2**-precision:
        return -precision
    if num > 2**precision:
        return precision

    k = 0
    greater = (num > 1)
    while True:
        if num > 1:
            if not greater:
                k = k + 1
                break
            num = num / 2
            k = k + 1
        elif num < 1:
            if greater:
                break
            num = num * 2
            k = k - 1
        else:
            break
    oarr = np.array(k, dtype=out_arr.dtype)
    oarr = np.array(pow(2, k), dtype=out_arr.dtype)
    out_arr.copyfrom(oarr)

def base2_round(scale):
    assert get_const_tuple(scale.shape) == (1, )
    base2_scale = tvm.extern((1, ), [scale],
        lambda ins, outs: tvm.intrin.call_packed("base2_round", ins[0], outs[0]),
        name='base2_round')
    # return topi.cast(base2_scale, 'int32')
    return topi.cast(base2_scale, 'float32')


@reg.register_compute("scale")
def compute_scale(attrs, inputs, _):
    mode = attrs["mode"]

    data = inputs[0]
    size = reduce(lambda x, y: x*y, get_const_tuple(data.shape))
    flatten_data = topi.reshape(data, (size, ))
    scale = topi.max(topi.abs(flatten_data), keepdims=True)
    if mode == 'real':
        return scale
    elif mode == 'base2':
        return base2_round(scale)
    else:
        raise ValueError



@reg.register_schedule("scale")
def schedule_scale(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])

@reg.register_compute("shrink_range")
def compute_shrink_range(attrs, inputs, _):
    out_dtype = attrs['out_type']
    data = inputs[0]
    scale = inputs[1]
    in_dtype = data.dtype

    size = reduce(lambda x, y: x*y, get_const_tuple(data.shape))
    flatten_data = topi.reshape(data, (size, ))

    quantile_scale = topi.max(topi.abs(flatten_data), keepdims=True)
    if scale.dtype == 'float32':
        quantile_scale = tvm.compute((1, ), lambda i: 1.0 * quantile_scale[0])
        shrink_data = tvm.compute(data.shape, lambda *i: data(*i).astype('float32') * (2**7 - 1.0) / quantile_scale[0])

    elif scale.dtype == 'int32':
        quantile_scale = base2_round(quantile_scale)
        shift = tvm.compute((1, ), lambda *i: (quantile_scale[0] - 7).astype(in_dtype))
        shrink_data = tvm.compute(data.shape, lambda *i: tvm.select(data(*i) < 0,
            - ((-data(*i)) >> shift[0]), data(*i) >> shift[0]))
    else:
        raise ValueError

    cliped_data = topi.clip(shrink_data, -127, 127)
    cast_data = tvm.compute(cliped_data.shape, lambda *i: tvm.select(cliped_data(*i) < 0,
        (cliped_data(*i) - 0.5).astype(out_dtype), (cliped_data(*i) + 0.5).astype(out_dtype)), name='cast')

    if scale.dtype == 'float32':
        upper_bound = (2**31 - 1) if in_dtype == "int32" else (2**15  - 1)
        updated_scale = tvm.compute((1, ), lambda *i: scale[0] * quantile_scale[0] / upper_bound)
    elif scale.dtype == 'int32':
        upper_bound = 31 if in_dtype == "int32" else 15
        updated_scale = tvm.compute((1, ), lambda *i: scale[0] + quantile_scale[0] - upper_bound)

    return [cast_data, updated_scale]

@reg.register_schedule("shrink_range")
def schedule_shrink_range(_, outs, target):
    return tvm.create_schedule([x.op for x in outs])
