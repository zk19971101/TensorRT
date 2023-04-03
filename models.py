import torch
import numpy as np
import tensorrt as trt
from torch import nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        z = F.softmax(y, dim=1)
        z = torch.argmax(z, dim=1)
        return y, z
    


def model_build(paraFile, network, inputTensor):
    para = np.load(paraFile)
    w = np.ascontiguousarray(para["conv1.weight"])
    b = np.ascontiguousarray(para["conv1.bias"])
    _0 = network.add_convolution_nd(inputTensor, 32, [5, 5], trt.Weights(w), trt.Weights(b))
    _0.padding_nd = [2, 2]
    _1 = network.add_activation(_0.get_output(0), trt.ActivationType.RELU)
    _2 = network.add_pooling_nd(_1.get_output(0), trt.PoolingType.MAX, [2, 2])
    _2.stride_nd = [2, 2]

    w = np.ascontiguousarray(para["conv2.weight"])
    b = np.ascontiguousarray(para["conv2.bias"])
    _3 = network.add_convolution_nd(_2.get_output(0), 64, [5, 5], trt.Weights(w), trt.Weights(b))
    _3.padding_nd = [2, 2]
    _4 = network.add_activation(_3.get_output(0), trt.ActivationType.RELU)
    _5 = network.add_pooling_nd(_4.get_output(0), trt.PoolingType.MAX, [2, 2])
    _5.stride_nd = [2, 2]

    _6 = network.add_shuffle(_5.get_output(0))
    _6.reshape_dims = (-1, 64 * 7 * 7)

    w = np.ascontiguousarray(para["fc1.weight"].transpose())
    b = np.ascontiguousarray(para["fc1.bias"].reshape(1, -1))
    _7 = network.add_constant(w.shape, trt.Weights(w))
    _8 = network.add_matrix_multiply(_6.get_output(0), trt.MatrixOperation.NONE, _7.get_output(0), trt.MatrixOperation.NONE)
    _9 = network.add_constant(b.shape, trt.Weights(b))
    _10 = network.add_elementwise(_8.get_output(0), _9.get_output(0), trt.ElementWiseOperation.SUM)
    _11 = network.add_activation(_10.get_output(0), trt.ActivationType.RELU)

    w = np.ascontiguousarray(para["fc2.weight"].transpose())
    b = np.ascontiguousarray(para["fc2.bias"].reshape(1, -1))
    _12 = network.add_constant(w.shape, trt.Weights(w))
    _13 = network.add_matrix_multiply(_11.get_output(0), trt.MatrixOperation.NONE, _12.get_output(0), trt.MatrixOperation.NONE)
    _14 = network.add_constant(b.shape, trt.Weights(b))
    _15 = network.add_elementwise(_13.get_output(0), _14.get_output(0), trt.ElementWiseOperation.SUM)

    _16 = network.add_softmax(_15.get_output(0))
    _16.axes = 1 << 1

    _17 = network.add_topk(_16.get_output(0), trt.TopKOperation.MAX, 1, 1 << 1)

    network.mark_output(_17.get_output(1))
    return network