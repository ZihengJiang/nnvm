# coding: utf-8
from __future__ import absolute_import
import numpy as np

import tvm
from tvm.contrib import graph_runtime
from collections import namedtuple

from . import graph as _graph
from . import compiler as _compiler
from .compiler.graph_util import infer_shape, infer_dtype
from .compiler.build_module import precompute_prune

_collect_internal_outputs = tvm.get_global_func("nnvm.quantization.CollectInternalOutputs")

CalibrationEntry = namedtuple("CalibrationEntry", ['min_value', 'max_value'])

def _base2_range(num, precision=32):
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
                return k + 1
            num = num / 2
            k = k + 1
        elif num < 1:
            if greater:
                return k
            num = num * 2
            k = k - 1
        else:
            return k

def execute_graph(module, inputs, oshapes, odtypes):
    module.set_input(**inputs)
    module.run()

    outs = []
    for i in range(len(oshapes)):
        arr = tvm.nd.empty(oshapes[i], dtype=odtypes[i])
        module.get_output(i, arr)
        outs.append(arr)

    return outs

def _shape_dtype_dict(inputs, params=None):
    ishapes = {k : v.shape for k, v in inputs.items()}
    idtypes = {k : v.dtype for k, v in inputs.items()}
    if params is not None:
        for key, param in params.items():
            ishapes[key] = param.shape
            idtypes[key] = param.dtype
    return ishapes, idtypes


def generate_entry(out, mode='full'):
    if mode in ['full', 'mean-max']:
        min_value = np.amin(out.asnumpy())
        max_value = np.amax(out.asnumpy())
        return {'min_value': min_value, 'max_value': max_value}
    if mode in ['partial', 'max-percentile']:
        return {'array': out.asnumpy()}

    raise ValueError


def get_bounds(records, mode='full', rate=None):
    def get_percentile(arr, rate):
        sorted_arr = np.sort(arr)
        neg_arr = sorted_arr[np.where(sorted_arr < 0)]
        if neg_arr.size == 0:
            lower_bound = 0
        else:
            lower_bound = np.percentile(neg_arr, (1-rate)*100)
        pos_arr = sorted_arr[np.where(sorted_arr > 0)]
        if pos_arr.size == 0:
            upper_bound = 0
        else:
            upper_bound = np.percentile(pos_arr, rate*100)
        return lower_bound, upper_bound

    if mode == 'full':
        print("{}\n".format(records[15]))
        lower_bound = min(entry['min_value'] for entry in records)
        upper_bound = max(entry['max_value'] for entry in records)
        return lower_bound, upper_bound
    if mode == 'mean-max':
        print("{}\n".format(records))
        lower_bound = reduce(lambda x, y: x + y, [entry['min_value'] for entry in records]) / len(records)
        upper_bound = reduce(lambda x, y: x + y, [entry['max_value'] for entry in records]) / len(records)
        return lower_bound, upper_bound

    if mode == 'partial':
        assert rate is not None
        arr = np.concatenate([entry['array'].flatten() for entry in records])
        lower_bound, upper_bound = get_percentile(arr, rate)

        min_value = min(np.amin(entry['array'].flatten()) for entry in records)
        max_value = max(np.amax(entry['array'].flatten()) for entry in records)
        print('min={}, max={}'.format(min_value, max_value))

        return lower_bound, upper_bound

    if mode == 'max-percentile':
        lbounds = []
        ubounds = []
        for entry in records:
            arr = entry['array']
            lower, upper = get_percentile(arr, rate)
            lbounds.append(lower)
            ubounds.append(upper)

        lower_bound = min(lbounds)
        upper_bound = max(ubounds)
        min_value = min(np.amin(entry['array'].flatten()) for entry in records)
        max_value = max(np.amax(entry['array'].flatten()) for entry in records)
        print('min={}, max={}'.format(min_value, max_value))
        return lower_bound, upper_bound


    raise ValueError


def collect_statistics(graph, dataset, params={}):
    mode = 'full'
    mode = 'mean-max'
    # mode = 'partial'
    # mode = 'max-percentile'
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)
    graph = graph.apply('SeparateBias')
    graph = _compiler.optimize(graph, ishapes, idtypes)
    graph, params = precompute_prune(graph, params)
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)

    # transform to statistic graph
    stats_graph = _collect_internal_outputs(graph);

    # build module
    stats_graph, lib, _ = _compiler.build(stats_graph.symbol, 'llvm', ishapes, idtypes)
    m = graph_runtime.create(stats_graph, lib, tvm.cpu(0))
    m.set_input(**params)

    # execute and collect stats
    records = {}  # dict from node name to list of entry
    out_names = stats_graph.symbol.list_output_names()
    _, oshapes = infer_shape(stats_graph, **ishapes)
    _, odtypes = infer_dtype(stats_graph, **idtypes)
    for inputs in dataset:
        outs = execute_graph(m, inputs, oshapes, odtypes)
        for i, out in enumerate(outs):
            key = out_names[i]
            entry = generate_entry(out, mode)
            if key in records:
                records[key].append(entry)
            else:
                records[key] = [entry]

    # analysis
    print('records:')
    base2_range = []
    for name in out_names:
        print('{}:'.format(name))
        lower_bound, upper_bound = get_bounds(records[name], mode, 0.95)
        k0 = _base2_range(lower_bound)
        k1 = _base2_range(upper_bound)
        print('lower={}, upper={}'.format(lower_bound, upper_bound))
        print("k={}, k0={}, k1={}".format(max(k0, k1), k0, k1))
        print('')
        base2_range.append(max(k0, k1))

    graph._set_json_attr("base2_range", base2_range, "list_int")
    return graph, params


def quantize(graph, debug=False):
    graph._set_json_attr("debug", int(debug), "int")
    qgraph = graph.apply('Quantize')
    return qgraph
