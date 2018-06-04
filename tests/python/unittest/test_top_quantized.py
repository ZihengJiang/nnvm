import nnvm.symbol as sym
from tvm.contrib import graph_runtime, util
import nnvm
import tvm
import numpy as np

target = 'llvm'
out_type = 'int32'

def test_quantized_dense():
    x_shape, y_shape = (4, 4), (2, 4)
    x = sym.ones_like(sym.zeros(shape=x_shape, name='x', dtype=out_type))
    y = sym.ones_like(sym.zeros(shape=y_shape, name='y', dtype=out_type))
    z = sym.quantized_dense(x, y, units=2, use_bias=False, name='fc', out_type=out_type)
    compute_graph = nnvm.graph.create(z)
    deploy_graph, lib, params = nnvm.compiler.build(
        compute_graph, target=target, shape={"x": x_shape, "y": y_shape, }, dtype=out_type)
    ctx = tvm.context(target, 0)
    module = graph_runtime.create(deploy_graph, lib, ctx)
    module.run()
    out = module.get_output(0, out=tvm.nd.empty((4,2), dtype=out_type))
    assert np.allclose(out.asnumpy(),np.ones((4,2))*4)

if __name__ == "__main__":
    test_quantized_dense()
