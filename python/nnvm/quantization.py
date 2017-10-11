# coding: utf-8
from __future__ import absolute_import
import tvm
from tvm.contrib import graph_runtime
import numpy as np
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


def collect_statistics(graph, dataset, params={}):
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)
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
            min_value = np.amin(out.asnumpy())
            max_value = np.amax(out.asnumpy())
            entry = CalibrationEntry(min_value, max_value)
            if key in records:
                records[key].append(entry)
            else:
                records[key] = [entry]

    # analysis
    base2_range = []
    for name in out_names:
        min_value = min(entry.min_value for entry in records[name])
        max_value = max(entry.max_value for entry in records[name])
        k0 = _base2_range(min_value)
        k1 = _base2_range(max_value)
        base2_range.append(max(k0, k1))

    graph._set_json_attr("base2_range", base2_range, "list_int")
    return graph, params


def quantize(graph, debug=False):
    graph._set_json_attr("debug", int(debug), "int")
    qgraph = graph.apply('Quantize')
    return qgraph
