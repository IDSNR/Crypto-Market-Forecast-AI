import numpy as np
import json
import math

def Leaky_ReLu(inputs, alpha):  # ReLu Activation a usar numpy
    # outputs = np.maximum(0, inputs)
    outputs = np.where(inputs > 0, inputs, inputs * alpha)
    return outputs

def Softmax_Activation(inputs):
    inputs -= inputs[np.argmax(inputs)]
    ex_inp = []
    for i in inputs:
        ex_inp.append(math.e ** i)
    probability = []
    for j in ex_inp:
        probability.append(j / (sum(ex_inp)))
    return probability

class Layer_Dense:
    def __init__(self, n_neurons, weights, biases, Leaky_ReLu_var=True, layer_after_input_var=False):
        self.n_neurons = n_neurons
        self.weights = np.array(weights)[:]
        self.biases = np.array(biases)[:]
        self.inputs = None
        self.outputs = None
        self.Leaky_ReLu = Leaky_ReLu_var
        self.inputs_data = layer_after_input_var
        self.List_Gradient_weights = None
        self.List_Gradient_bias = None
        self.real_weights = np.array(weights)[:]
        self.real_biases = np.array(biases)[:]

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        dot_product = np.dot(self.weights.T, inputs)
        self.outputs = dot_product + self.biases
        self.outputs = np.array(self.outputs)



crypto_name = "BTC"  # qualquer cryptoserve, desde que tenhas uma database dela

X = []
y_true = []

try:
    with open("D:\Python\Databases\database_" + str(crypto_name) + "_T.json", "r") as file:
        jr = json.load(file)
        for i in range(2, 500, 2):
            i = str(i)
            prices = jr[i]["Prices"]
            volume = jr[i]["Volume"]
            RSI = jr[i]["RSI"]
            Market_cap = jr[i]["Market_cap"]
            market_cap_TVL = jr[i]["MC/TVL"]
            outputs = jr[i]["Price_12_hours"]
            inputs_a = prices + volume + RSI + Market_cap + market_cap_TVL
            if len(inputs_a) == 202:
                X.append(inputs_a)
                if outputs > 100:  # mudar dependendo de preço de crypto
                    y_true.append([1, 0])
                else:
                    y_true.append([0, 1])
except Exception as e:
    print("An error occured:", e)


with open("D:\Python\Databases\Databases Weights And Biases\Database_weights_and_biases_"+str(crypto_name)+"_T.json", "r") as file:
    load = json.load(file)
    weigths_list_of_matrises_for_continuation = load["Weights"]
    biases_matrix_continuation = load["Biases"]
    best_set_of_biases_yet = load["Biases"]
    best_set_of_weights_yet = load["Weights"]
    loss_b = load["Loss"]
    neural_network_layers_and_its_neurons = load["Neurons_Positions"]
    Alpha_Leaky_ReLu = load["Alpha_LR"]

neural_network_layers = []
soma_accur = 0
y_pred = []

if __name__ == "__main__":
    for index, neurons_layer in enumerate(neural_network_layers_and_its_neurons[:-1]):
        if index == 0:
            pass
        elif index == 1:
            neural_network_layers.append(Layer_Dense(neurons_layer, weigths_list_of_matrises_for_continuation[index - 1], biases_matrix_continuation[index - 1], layer_after_input_var=True))
        else:
            neural_network_layers.append(Layer_Dense(neurons_layer, weigths_list_of_matrises_for_continuation[index - 1], biases_matrix_continuation[index - 1]))
    neural_network_layers.append(Layer_Dense(neural_network_layers_and_its_neurons[-1], weigths_list_of_matrises_for_continuation[-1], biases_matrix_continuation[-1], Leaky_ReLu_var=False))

    for a, i in enumerate(X):

        current_input = i

        for a, layer in enumerate(neural_network_layers):
            layer.forward(current_input)
            if a != (len(neural_network_layers) - 1):
                current_input = Leaky_ReLu(layer.outputs, Alpha_Leaky_ReLu)
            if a == (len(neural_network_layers) - 1):
                predicted_output = Softmax_Activation(layer.outputs)

        y_pred.append(predicted_output)

def accur(y_true, y_pred):
    soma = 0
    for i, j in zip(y_true, y_pred):
        if np.argmax(i) == np.argmax(j):
            soma += 1
    return soma/len(y_true)

print(accur(y_true, y_pred))










