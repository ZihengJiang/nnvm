# coding: utf-8
from __future__ import absolute_import
import numpy as np
import scipy
import scipy.stats
import math
import random

import tvm
from tvm.contrib import graph_runtime
from collections import namedtuple

from . import compiler as _compiler
from .compiler import graph_attr
from .compiler.graph_util import infer_shape, infer_dtype
from .compiler.build_module import precompute_prune
from .model import graph_execute, predict, evaluate, _shape_dtype_dict

_collect_internal_outputs = tvm.get_global_func("nnvm.quantization.CollectInternalOutputs")

CalibrationEntry = namedtuple("CalibrationEntry", ['min_value', 'max_value'])


def collect_statistics(graph, params, dataset):
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)

    # optimize
    graph = graph.apply('SeparateBias')
    graph = graph_attr.set_shape_inputs(graph, ishapes)
    graph = graph.apply(["InferShape", "SimplifyInference"])
    graph = graph_attr.set_shape_inputs(graph, ishapes)
    graph = graph.apply(["InferShape", "FoldScaleAxis"])
    graph, params = precompute_prune(graph, params)
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)

    # transform to statistic graph
    stats_graph = _collect_internal_outputs(graph);

    # build module
    stats_graph, lib, _ = _compiler.build(stats_graph.symbol, "llvm", ishapes, idtypes)
    m = graph_runtime.create(stats_graph, lib, tvm.cpu(0))
    m.set_input(**params)

    # execute and collect stats
    records = {}  # dict from node name to list of entry
    out_names = stats_graph.symbol.list_output_names()
    _, oshapes = infer_shape(stats_graph, **ishapes)
    _, odtypes = infer_dtype(stats_graph, **idtypes)
    for inputs in dataset:
        outs = graph_execute(m, inputs, oshapes, odtypes)
        for i, out in enumerate(outs):
            key = out_names[i]
            min_value = np.amin(out.asnumpy())
            max_value = np.amax(out.asnumpy())
            entry = {'min_value': min_value, 'max_value': max_value}
            if key in records:
                records[key].append(entry)
            else:
                records[key] = [entry]

    # analysis
    # print('records:')
    base2_range = []
    for name in out_names:
        # print('{}:'.format(name))
        # for entry in records[name]:
        #     print("{}, {}".format(entry['min_value'], entry['max_value']))
        lower_bound = min(entry['min_value'] for entry in records[name])
        upper_bound = max(entry['max_value'] for entry in records[name])
        eps = pow(2, -30)
        k0 = int(math.ceil(math.log(abs(lower_bound) + eps, 2)))
        k1 = int(math.ceil(math.log(abs(upper_bound) + eps, 2)))
        base2_range.append(max(k0, k1))

    stats = base2_range
    return graph, params, stats

def calibrate(graph, params, dataset, labels, stats):
    BOUND = 0.98
    MAX_ROUND = 10
    DISTURBANCE = [-2, -1, -0, 1, 2]

    best = stats
    best_rate = 0.01
    round_cnt = 0

    while best_rate <= BOUND and round_cnt < MAX_ROUND:
        print('\nround {0}'.format(round_cnt))
        for i in range(len(stats)):
            random.shuffle(DISTURBANCE)
            for disturbance in DISTURBANCE:
                candid = best
                candid[i] = candid[i] + disturbance
                qgraph = quantize(graph, candid)
                preds = predict(qgraph, params, dataset)[-1]
                rate = evaluate(preds, labels)
                print('candid: {0}'.format(candid))
                print('rate: {0}'.format(rate))
                if rate >= best_rate:
                    best = candid
                    best_rate= rate
        round_cnt += 1

    threshold = best
    return threshold

def quantize(graph, threshold):
    graph._set_json_attr("threshold", threshold, "list_int")
    qgraph = graph.apply("Quantize")
    return qgraph
