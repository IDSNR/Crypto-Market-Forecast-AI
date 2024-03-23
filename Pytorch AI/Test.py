import torch
import torch.nn
import numpy as np
import math

def Softmax_Activation(inputs):
    inputs -= inputs[np.argmax(inputs)]
    ex_inp = []
    for i in inputs:
        ex_inp.append(math.e ** i)
    probability = []
    for j in ex_inp:
        probability.append(j / (sum(ex_inp)))
    return probability


o = torch.tensor([1, 2])
npa = np.array([1, 2])
print(Softmax_Activation(npa))
print(torch.nn.softmax(o))