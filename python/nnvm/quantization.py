# coding: utf-8
from __future__ import absolute_import
import tvm
import nnvm.graph as _graph
import nnvm.compiler as _compiler
from tvm.contrib import graph_runtime
from nnvm.compiler.graph_util import infer_shape
import numpy as np
from collections import namedtuple

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
                return k
            num = num / 2
            k = k + 1
        elif num < 1:
            if greater:
                return k
            num = num * 2
            k = k - 1
        else:
            return k

def execute_graph(module, inputs, oshapes):
    module.set_input(**inputs)
    module.run()

    outs = []
    for i in range(len(oshapes)):
        arr = tvm.nd.empty(oshapes[i], dtype='float32')
        module.get_output(i, arr)
        outs.append(arr)

    return outs


def collect_statistics(graph, dataset, params=None):
    # transform graph
    graph = _collect_internal_outputs(graph);
    ishapes = {k : v.shape for k, v in dataset[0].items()}
    _, oshapes = infer_shape(graph, **ishapes)
    graph = _compiler.optimize(graph, ishapes)

    graph_, lib, _ = _compiler.build(graph.symbol, 'llvm', ishapes)
    m = graph_runtime.create(graph_, lib, tvm.cpu(0))
    if params is not None:
        m.set_input(**params)

    # execute and collect record
    out_names = graph.symbol.list_output_names()
    records = {}  # dict from node name to list of entry
    for inputs in dataset:
        outs = execute_graph(m, inputs, oshapes)
        for i, out in enumerate(outs):
            key = out_names[i]
            min_value = np.amin(out.asnumpy())
            max_value = np.amax(out.asnumpy())
            entry = CalibrationEntry(min_value, max_value)
            if key in records:
                records[key].append(entry)
            else:
                records[key] = [entry]

    base2_range = []
    for name in out_names:
        min_value = min(entry.min_value for entry in records[name])
        max_value = max(entry.max_value for entry in records[name])
        k0 = _base2_range(min_value)
        k1 = _base2_range(max_value)
        base2_range.append(max(k0, k1))

    graph._set_json_attr("base2_range", base2_range, "list_int")
    return graph
