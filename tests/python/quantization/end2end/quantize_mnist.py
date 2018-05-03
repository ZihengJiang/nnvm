import argparse
import numpy as np
import tvm
import nnvm
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr
from nnvm.compiler.build_module import precompute_prune
from nnvm.quantization import *

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import get_data, get_model, evaluate_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['mlp', 'conv', 'bn', 'res_block'],
    help="The model type.")
parser.add_argument('--num-calibration-set', type=int, default=50,
    help="Number of calibration set.")
args = parser.parse_args()

net_name = args.model
data_name = 'mnist'
num_calibration_set = args.num_calibration_set

# prepare the model
net = get_model(net_name)
print('net: ')
print(net)
print('')
sym, params = nnvm.frontend.from_mxnet(net)
# sym = nnvm.sym.softmax(sym)
print('params: {0}'.format(params.keys()))
print('')

# prepare the dataset
train_data, test_data = get_data(data_name)
dataset = []
labels = []
for batch_data, batch_label in test_data:
    dataset.append({'data': batch_data.asnumpy()})
    labels.append(batch_label.asnumpy())

# quantization
graph = nnvm.graph.create(sym)
graph, params, stats = collect_statistics(graph, params, dataset[:num_calibration_set])
threshold = stats
threshold = calibrate(graph, params, dataset[:num_calibration_set], labels[:num_calibration_set], stats)
qgraph = quantize(graph, threshold)
print(qgraph.symbol.debug_str())


# execute quantized graph
egraph = qgraph
shape_dict = {k: v.shape for k, v in params.items()}
dtype_dict = {k: v.dtype for k, v in params.items()}
for key, data in dataset[0].items():
    shape_dict[key] = data.shape
    dtype_dict[key] = data.dtype

egraph, lib, params = nnvm.compiler.build(egraph.symbol, 'llvm', shape_dict, dtype_dict, params)
m = graph_runtime.create(egraph, lib, tvm.cpu(0))
m.set_input(**params)

print('evaluating...')
num = len(dataset)
evaluate_accuracy(m, egraph, dataset[:num], labels[:num], params)
