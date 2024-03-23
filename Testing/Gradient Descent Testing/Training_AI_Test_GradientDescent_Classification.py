import numpy as np
import Functions as F
import math
import json
import time
import termcolor as color
import backpropagation_Functions as bF

crypto_name = "BTC"

inicial_inputs = []

# 12 horas depois do último preço
# 36 horas de dados, no total, para treino
expected_outputs = []

try:
    with open("D:\Python\Python_Crypto_AI\Testing\database_"+str(crypto_name)+"_T.json", "r") as file:
        jr = json.load(file)
        for i in range(1, 501, 2):
            i = str(i)
            prices = jr[i]["Prices"]
            volume = jr[i]["Volume"]
            RSI = jr[i]["RSI"]
            Market_cap = jr[i]["Market_cap"]
            market_cap_TVL = jr[i]["MC/TVL"]
            outputs = jr[i]["Price_12_hours"]
            inputs_a = prices + volume + RSI + Market_cap + market_cap_TVL
            inicial_inputs.append(inputs_a)
            if outputs > 100: #mudar dependendo do preço normal de cada crypto (aproximadamente 0.1 - 0-2 % do preço dela, para ter em conta transaction fees)
                expected_outputs.append([1, 0])
            else:
                expected_outputs.append([0, 1])
except Exception as e:
    print("An error occured:", e)

inicial_inputs = (np.array(inicial_inputs) * 0.01).tolist()

X = inicial_inputs[:]

Loss = 1
accuracy_b = 0
neural_network_layers = []
a3 = 1
a8 = False
a12 = 0
a15 = True
Epoch_or_not = False

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
        probability.append(j / (sum(ex_inp)/len(ex_inp)))
    return probability
def Leaky_ReLu_derivative(inputs, alpha):
    return np.where(inputs > 0, inputs, inputs / alpha)


class Layer_Dense:
    def __init__(self, n_neurons, weights, biases, Leaky_ReLu_var=True):
        self.n_neurons = np.array(n_neurons)
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.inputs = None
        self.outputs = None
        self.Leaky_ReLu = Leaky_ReLu_var

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.outputs = np.dot(self.weights.T, inputs) + self.biases
        self.outputs = np.array(self.outputs)

    def backpropagation(self, gradient, learning_rate, alpha):
        gradient_to_pass = 0
        if self.Leaky_ReLu:
            mean_gradient = sum(gradient)/len(gradient)

            gradient_to_pass = bF.media_matrix(self.weights) * Leaky_ReLu_derivative(self.inputs, alpha) * mean_gradient

            gradient_weights = np.array(bF.listas_to_matrtix(self.inputs, gradient))
            gradient_bias = np.array(gradient * (sum(self.biases) / len(self.biases)))

            self.weights -= learning_rate * gradient_weights
            self.biases -= learning_rate * gradient_bias

        if not self.Leaky_ReLu:
            mean_of_gradient = sum(gradient)/len(gradient)

            inputs_no_activation = Leaky_ReLu_derivative(self.inputs, alpha)

            gradient_to_pass = inputs_no_activation * bF.media_matrix(self.weights) * mean_of_gradient * (sum(self.outputs)/len(self.outputs))

            outer_of_inputs_and_outputs = bF.listas_to_matrtix(inputs_no_activation, self.outputs)

            gradient_weights = mean_of_gradient * np.array(outer_of_inputs_and_outputs)
            self.weights -= learning_rate * gradient_weights

            gradient_bias = (self.outputs * gradient) * (sum(self.biases) / len(self.biases))
            gradient_bias -= learning_rate * gradient_bias

        return gradient_to_pass


input_layer_n_neurons = len(inicial_inputs[0])

learning_rate = 0.01
scale = 10
Epoch_number = 25
begginer_number = 1e-5
scale_gradient_of_loss = 1
Tottaly_Random = False
longer_slope_learning_rate_opt = 2
Batch_size = 10000
Loss_Threshold = 1e-3
Alpha_Leaky_ReLu = 0.001
neural_network_layers_and_its_neurons = [input_layer_n_neurons, 10, 10, 5, 3, 2]
iteration = 5


weigths_list_of_matrises_for_continuation = F.random_weights_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)
biases_matrix_continuation = F.random_bias_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)




'''
with open("Database_weights_and_biases_"+str(iteration)+""+str(crypto_name)+".json", "r") as file:
   load = json.load(file)
   weigths_list_of_matrises_for_continuation = load["Weights"]
   biases_matrix_continuation = load["Biases"]
   best_set_of_biases_yet = load["Biases"]
   best_set_of_weights_yet = load["Weights"]
   loss_b = load["Loss"]
'''



if Epoch_number != None:
    Epoch_or_not = True
if Epoch_or_not:
    a15 = (a12 <= Epoch_number)

lists_opt_weights = []
lists_opt_biases = []
loss_opt = []
accuracy_opt = []

while Loss > Loss_Threshold and a15:

    if Epoch_or_not:
        a15 = (a12 >= Epoch_number)

    if a12 > 0:
        loss_opt.append(loss_b)
        accuracy_opt.append(accuracy_b)

    if Tottaly_Random:
        weigths_list_of_matrises_for_continuation = F.random_weights_for_Inicialization_and_continuation(
            neural_network_layers_and_its_neurons, begginer_number, scale)
        biases_matrix_continuation = F.random_bias_for_Inicialization_and_continuation(
            neural_network_layers_and_its_neurons, begginer_number, scale)

    elif a12 > 0 and (not Tottaly_Random):
        weigths_list_of_matrises_for_continuation = F.matriz_muda_pesos_og(best_set_of_weights_yet, scale)
        biases_matrix_continuation = F.muda_biases_og(best_set_of_biases_yet, scale)


    for a in range(1, len(neural_network_layers_and_its_neurons) - 1):
        if a != len(neural_network_layers_and_its_neurons) - 1:
            neural_network_layers.append(Layer_Dense(neural_network_layers_and_its_neurons[a], weigths_list_of_matrises_for_continuation[a - 1], biases_matrix_continuation[a - 1]))
        if a == len(neural_network_layers_and_its_neurons) - 2:
            neural_network_layers.append(Layer_Dense(neural_network_layers_and_its_neurons[a + 1], weigths_list_of_matrises_for_continuation[a],biases_matrix_continuation[a], False))

    a12 += 1

    a8 = True

    opt_b = []
    opt_w = []

    for i in neural_network_layers:
        opt_w.append(i.weights)
        opt_b.append(i.biases)

    lists_opt_biases.append(opt_b)
    lists_opt_weights.append(opt_w)

    if Epoch_or_not:
        color.cprint("Epoch number:", "light_yellow"), color.cprint(a12, "light_yellow")
    else:
        color.cprint("Mais uma", "light_yellow")

    loss = []
    accuracy = []

    for i in range(0, Batch_size):

        real_outputs = []

        for a21, i in enumerate(X):

            current_input = i

            # Forward pass through each layer
            for a, layer in enumerate(neural_network_layers):
                layer.forward(current_input)
                if a != (len(neural_network_layers) - 1):
                    current_input = Leaky_ReLu(layer.outputs, Alpha_Leaky_ReLu)
                if a == (len(neural_network_layers) - 1):
                    current_input = layer.outputs

            loss_gradient = F.gradient_of_loss_multiple_output_neurons(expected_outputs[a21], current_input, scale_gradient_of_loss)

            current_gradient = loss_gradient
            for layer in reversed(neural_network_layers):
                current_gradient = layer.backpropagation(current_gradient, learning_rate, Alpha_Leaky_ReLu)

            current_input = Softmax_Activation(current_input)
            real_outputs.append(current_input)

        loss_a, accuracy_a = F.accuracy_and_or_loss_in_one_output_Softmax(expected_outputs, real_outputs)

        accuracy.append(accuracy_a)
        loss.append(loss_a) #mudar

        if a12 == 1:
            loss_b = loss_a + 1e10

        Loss = (sum(loss) / len(loss))


        opt_w2 = []
        opt_b2 = []

        loss_b = loss_a
        accuracy_b = accuracy_a

        color.cprint("Accuracy", "light_green"), color.cprint(accuracy_a, "light_green")
        print("\n")
        color.cprint("Loss:", "light_green"), color.cprint(accuracy_a, "light_green")

        for i in neural_network_layers:
            opt_w2.append(i.weights)
            opt_b2.append(i.biases)

        best_set_of_weights_yet = opt_w2[:]
        best_set_of_biases_yet = opt_b2[:]

        if a8:
            obj = {
                "Date": int(time.time()),
                "Weights": list(best_set_of_weights_yet),
                "Biases": list(best_set_of_biases_yet),
                "Accuracy:": accuracy_b,
                "Loss": loss_b,
                "Neurons_list": neural_network_layers_and_its_neurons
            }
            try:
                with open("Database_weights_and_biases_5" + str(crypto_name) + ".json", "w") as file:
                    json.dump(obj, file, indent=2)
                print("Data written to database.json successfully.")
            except Exception as e:
                print("An error occurred:", e)

        a8 = False

        learning_rate_usb = learning_rate * F.learning_rate_opt(a12, Batch_size, 2)

a = F.retorna_menor_indice_lista(loss_opt, 1)

best_set_of_biases_yet = lists_opt_biases[a]
best_set_of_weights_yet = lists_opt_weights[a]
loss_b = loss_opt[a]
accuracy_b = accuracy_opt[a]

obj = {
    "Date": int(time.time()),
    "Weights": list(best_set_of_weights_yet),
    "Biases": list(best_set_of_biases_yet),
    "Accuracy:": list(accuracy_b),
    "Loss": list(loss_b),
    "Neurons_list": neural_network_layers_and_its_neurons
}

try:
    with open("Database_weights_and_biases_4" + str(crypto_name) + ".json", "w") as file:
        json.dump(obj, file, indent=2)
        print("Data written to database.json successfully.")
except Exception as e:
    print("An error occurred:", e)