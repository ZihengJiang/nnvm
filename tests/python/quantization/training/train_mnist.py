import argparse
import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import get_data, get_net, get_model

mx.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=['mlp', 'conv', 'bn', 'res_block'],
    help="The model type.")
parser.add_argument('--num-epoch', type=int, default=2,
    help="Number of epoch during training.")
parser.add_argument('--inference', type=bool, default=False,
    help="Use saved params for inference.")
parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
    help='The device type')
args = parser.parse_args()

ctx = mx.gpu(0) if args.device == 'gpu' else mx.cpu()
net_name = args.model
data_name = 'mnist'

train_data, test_data = get_data(data_name)

net = get_net(net_name)
print(net)
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})


def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

smoothing_constant = .01

def train():
    for e in range(args.num_epoch):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        test_accuracy = evaluate_accuracy(test_data, net, ctx)
        train_accuracy = evaluate_accuracy(train_data, net, ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

if args.inference:
    net = get_model(net_name, ctx)
    test_accuracy = evaluate_accuracy(test_data, net, ctx)
    print('test acc %s' % (test_accuracy*100))
else:
    train()
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    filename = "checkpoints/{}_{}.params".format(data_name, net_name)
    print('save model at {}'.format(filename))
    net.save_params(filename)
