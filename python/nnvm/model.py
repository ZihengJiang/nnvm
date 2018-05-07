# coding: utf-8
from __future__ import absolute_import
import numpy as np

import tvm
from tvm.contrib import graph_runtime

from . import graph as _graph
from . import compiler as _compiler
from .compiler.graph_util import infer_shape, infer_dtype

def graph_execute(module, inputs, oshapes, odtypes):
    """graph execution on a batch data
        (num_output, (batch_size, output_dim...))"""
    module.set_input(**inputs)
    module.run()

    outs = []
    for i in range(len(oshapes)):
        arr = tvm.nd.empty(oshapes[i], dtype=odtypes[i])
        module.get_output(i, arr)
        outs.append(arr)

    return outs


def predict(graph, params, dataset, batch_size=1,
            target='llvm', context=tvm.cpu(0)):
    """return outputs of a dataset
        (num_output, num_batch, (batch_size, output_dim...))"""
    ishapes, idtypes = _shape_dtype_dict(dataset[0], params)
    egraph, lib, _ = _compiler.build(graph.symbol, target, ishapes, idtypes)
    mod = graph_runtime.create(egraph, lib, context)
    mod.set_input(**params)
    _, oshapes = infer_shape(egraph, **ishapes)
    _, odtypes = infer_dtype(egraph, **idtypes)
    num_output = len(oshapes)

    outputs = [[] for i in range(num_output)]
    for inputs in dataset:
        outs = graph_execute(mod, inputs, oshapes, odtypes)
        for i in range(num_output):
            outputs[i].append(outs[i])

    return outputs


def evaluate(preds, labels):
    count = 0
    for idx, (pred, label) in enumerate(zip(preds, labels)):
        pred = pred.asnumpy()
        pred = np.argmax(pred[0])
        label = int(label[0])
        if pred == label:
            count += 1
        rate = float(count) / (idx+1)
    rate = float(count) / len(preds)
    return rate


def _shape_dtype_dict(inputs, params=None):
    ishapes = {k : v.shape for k, v in inputs.items()}
    idtypes = {k : v.dtype for k, v in inputs.items()}
    if params is not None:
        for key, param in params.items():
            ishapes[key] = param.shape
            idtypes[key] = param.dtype
    return ishapes, idtypes
