import torch
import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork:
    def __init__(self):
        # layer size = [10, 8, 8, 4]
        # 初始化所需参数   
        pass

    def feedforward(self, x):
        # 前向传播
        pass

    def torch_grad(self, x, y):
        pass


    def backpropagation(self, inputs, true_labels, epochs = 10, lr = 0.05):
        pass

def main():
    # 设置随机种子,保证结果的可复现性
    np.random.seed(1)
    nn = NeuralNetwork()

    # 生成数据
    inputs = np.random.rand(100, 5)

    # 生成one-hot标签
    true_labels = np.eye(3)[np.random.randint(0, 3, size=(1, 100))] # one-hot label

    # train
    nn.backpropagation(inputs, true_labels, 100)



main()