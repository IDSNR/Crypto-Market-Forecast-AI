import numpy as np
import Functions as F
import math
import json
import random
import time
import termcolor as color
import dropout_Functions as dF

crypto_name = "BTC"  # qualquer cryptoserve, desde que tenhas uma database dela


X = []
y_true = []

try:
    with open("D:\Python\Databases\database_" + str(crypto_name) + "_T.json", "r") as file:
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
            if len(inputs_a) == 202:
                X.append(inputs_a)
                if outputs > 100:  # mudar dependendo de preço de crypto
                    y_true.append([1, 0])
                else:
                    y_true.append([0, 1])
except Exception as e:
    print("An error occured:", e)

# if not the first iteration of trainig
'''
with open("D:\Python\Databases\Databases Weights And Biases\Database_weights_and_biases_"+str(crypto_name)+"_T.json", "r") as file:
    load = json.load(file)
    weigths_list_of_matrises_for_continuatio = load["Weights"]
    biases_matrix_continuatio = load["Biases"]
    best_set_of_biases_yet = load["Biases"]
    best_set_of_weights_yet = load["Weights"]
    loss_b = load["Loss"]
Using_Previous_Data = True

weigths_list_of_matrises_for_continuation = []
biases_matrix_continuation = []
for i in reversed(weigths_list_of_matrises_for_continuatio):
    weigths_list_of_matrises_for_continuation.append(i)
for j in reversed(biases_matrix_continuatio):
    biases_matrix_continuation.append(j)
'''

# initialize some variables
Loss = 1
accuracy_a = -10
loss_a = 10
accuracy_b = 0
neural_network_layers = []
Epoch_number_Count = 0
epoch_bigger_than_epoch_number = True
Epoch_or_not = False
put_to_database = False
input_layer_n_neurons = len(X[0])
lists_opt_weights = []
lists_opt_biases = []
loss_opt = []
accuracy_opt = []
opt_b = []
opt_w = []
opt_accuracy = []
opt_loss = []
Var_So_Not_As_Much_Dropout = True

def Soma_Listas_Matrix(matrix):
    matrix = np.array(matrix)
    current_list = matrix[0]
    for i in matrix[1:]:
        current_list += i
    return current_list

def Leaky_ReLu(inputs, alpha):  # ReLu Activation a usar numpy
    # outputs = np.maximum(0, inputs)
    outputs = np.where(inputs > 0, inputs, inputs * alpha)
    return outputs


def gradient_of_loss(true_value, predicted_value, scale=1):
    gradient = []
    for i, j in zip(predicted_value, true_value):
        gradient.append((i - j)*scale)
    return gradient

def Loss_Function(y_true, y_pred):
    loss = []
    y_pred = np.array(y_pred)
    y_pred += 1e-7
    for true, j in zip(y_true, y_pred):
        true_choice = np.argmax(true)
        loss_entropy = -(math.log(j[true_choice]))
        loss.append(loss_entropy)
    loss_re = np.sum(loss)
    soma = 0
    for tr, pr in zip(y_true, y_pred):
        if np.argmax(tr) == np.argmax(pr):
            soma += 1
    accuracy = soma / len(y_true)
    return loss_re, accuracy


def Softmax_Activation(inputs):
    inputs -= inputs[np.argmax(inputs)]
    ex_inp = []
    for i in inputs:
        ex_inp.append(math.e ** i)
    probability = []
    for j in ex_inp:
        probability.append(j / (sum(ex_inp)))
    return probability


def Leaky_ReLu_derivative(inputs, alpha):
    return np.where(inputs > 0, inputs, inputs / alpha)


class Layer_Dense:
    def __init__(self, n_neurons, weights, biases):
        self.n_neurons = n_neurons
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.outputs = None
        self.best_biases = np.array(biases)
        self.best_weights = np.array(weights)

    def forward(self, inputs):
        dot_product = np.dot(self.weights.T, inputs)
        self.outputs = dot_product + self.biases
        self.outputs = np.array(self.outputs)

    def RL_Backpropagation(self, weights_changed, biases_changed):
        self.weights = weights_changed
        self.bias = biases_changed

    def RL_Changes(self):
        self.best_weights = self.weights
        self.best_biases = self.biases


# relevant and important variables
scale = 1  # Scale of new biases and weights, initial and for each epoch as well
Epoch_number = 25  # Number of tries to get the best neural netwrok, if you only want to have it stop training after the loss is less than loss threshold set it to None
begginer_number = 1e-5  # Approximate value of initial weights and biases
Tottaly_Random = True  # Is the new weigths in the new epoch decided randomly or with base on the best set of weights and Biases yet found?
longer_slope_learning_rate_opt = 2  # slope of sigmoid function that makes the learning rate adjust over time
N_iterations_per_Epoch = 100000  # number of iterations per epoch
Loss_Threshold = 1e-3  # How small must the loss be in order for it to stop training
Alpha_Leaky_ReLu = 0.001  # Alpha of leaky ReLu
neural_network_layers_and_its_neurons = [input_layer_n_neurons, 50, 30, 15, 7, 5, 2]  # Number of neurons per layer, change implies a from-ground training
Using_Previous_Data = False  # Put this True if using previous data
Batch_size_training = len(X[0])  # Batch size of training per iteration(if you want to do Batch Gradient Descent just put in len(X[0]), and if you want to do Stochastic Gradient Descent justput in one (1))
Mini_Batch_repetitions = 10  # how many times does the neural network get trained per iteration, if you want to just do one per iteraation just put Mini_Batch_repetitions = 1


while Loss > Loss_Threshold and epoch_bigger_than_epoch_number:

    # initializing some lists for optimization
    # for optimization of weigths
    opt_w2 = []
    # for optimization of biases
    opt_b2 = []
    # for optimization of loss
    opt_loss = []
    # for optimization of accuracy
    opt_accuracy = []

    if Epoch_number_Count > 0:
        for i in neural_network_layers:
            opt_w2.append(i.best_weights)
            opt_b2.append(i.best_biases)
        opt_w.append(opt_w2)
        opt_b.append(opt_b2)
        opt_accuracy.append(accuracy_a)
        opt_loss.append(loss_b)

    if Epoch_number != None:
        epoch_bigger_than_epoch_number = (Epoch_number_Count <= Epoch_number)

    if Epoch_number_Count > 0:
        loss_opt.append(loss_a)
        accuracy_opt.append(accuracy_a)

    # Please if you are going to useprevious weights and biases put Using_Previous_Data True
    if Tottaly_Random and not Using_Previous_Data:
        weigths_list_of_matrises_for_continuation = F.random_weights_for_Inicialization_and_continuation(
            neural_network_layers_and_its_neurons, random.random() * begginer_number, scale)
        biases_matrix_continuation = F.random_bias_for_Inicialization_and_continuation(
            neural_network_layers_and_its_neurons, random.random() * begginer_number, scale)

    # Changing weights and biases based on scale if not tottally random
    elif not Tottaly_Random and not Using_Previous_Data:
        weigths_list_of_matrises_for_continuation = F.matriz_muda_pesos_og(opt_w2, (scale * 100))
        biases_matrix_continuation = F.muda_biases_og(opt_b2, (scale * 100))


    #Building neural netwrok initial infrestracture
    for index, neurons_layer in enumerate(neural_network_layers_and_its_neurons[:-1]):
        if index == 0:
            pass
        elif index == 1:
            neural_network_layers.append(Layer_Dense(neurons_layer, weigths_list_of_matrises_for_continuation[index - 1], biases_matrix_continuation[index - 1]))
        else:
            neural_network_layers.append(Layer_Dense(neurons_layer, weigths_list_of_matrises_for_continuation[index - 1], biases_matrix_continuation[index - 1]))
    neural_network_layers.append(Layer_Dense(neural_network_layers_and_its_neurons[-1], weigths_list_of_matrises_for_continuation[-1], biases_matrix_continuation[-1]))

    # Counting Epochs
    Epoch_number_Count += 1

    # Printing Epoch Number
    color.cprint("Epoch number:", "light_yellow"), color.cprint(Epoch_number_Count, "light_yellow")

    # starting Batch Training
    for i in range(N_iterations_per_Epoch):

        # initiating soma variables
        expected_outs_all = []
        real_outputs_all = []

        # incorporate code from before
        for m in range(Mini_Batch_repetitions):

            # initializing soma variables
            real_outputs = []
            loss_gradient = []

            # Sorting Lists to avoid local minima
            y_true2, X2 = F.Baralha_Listas(y_true, X, Batch_size_training)

            for current_index, batch in enumerate(X2):

                # input = current input
                current_input = batch

                # forward pass
                for a, layer in enumerate(neural_network_layers):
                    layer.forward(current_input)
                    if a != (len(neural_network_layers) - 1):
                        current_input = Leaky_ReLu(layer.outputs, Alpha_Leaky_ReLu)
                    if a == (len(neural_network_layers) - 1):
                        predicted_output = Softmax_Activation(layer.outputs)

                # Pôr output da rede neuronal para calcular perda
                real_outputs.append(predicted_output)

            # Pôr os que estão agora a ser usados para assim a ordem coincidir
            expected_outs_all += y_true2
            real_outputs_all += real_outputs

        # Compute loss and accuracy
        loss_a, accuracy_a = Loss_Function(expected_outs_all, real_outputs_all)

        if i == 0:
            accuracy_b = accuracy_a + 1

        if accuracy_a > accuracy_b:
            accuracy_b = accuracy_a
            for layer2 in neural_network_layers:
                layer2.RL_Changes()
            print("New best found!")
            put_to_database = True

        for layer in neural_network_layers:
            layer.RL_Backpropagation(F.muda_pesos_RL(layer.best_weights, scale), F.muda_biases_RL(layer.best_biases, scale))

        # put information in database.json
        if put_to_database:

            # initializing parameters
            opt_w3 = []
            opt_b3 = []

            # getting weights and biases
            for layer2 in neural_network_layers:
                opt_w3.append(layer2.weights.tolist())
                opt_b3.append(layer2.biases.tolist())

            # setting up object to put in .json
            object_opt = {
                "Weights": opt_w3,
                "Biases": opt_b3,
                "Accuracy": accuracy_a,
                "Loss": loss_a,
                "Neurons_Positions": neural_network_layers_and_its_neurons
            }

            # opening file
            try:
                with open("D:\Python\Databases\Databases Weights And Biases\Database_weights_and_biases_" + str(
                        crypto_name) + "_T2.json", "w") as file:
                    json.dump(object_opt, file, indent=2)
            except Exception as e:
                print("An error occured:", e)

        #Printing Results
        color.cprint("Loss:", "light_red"), color.cprint(loss_a, "light_red")
        color.cprint("Accuracy:", "light_green"), color.cprint(accuracy_a, "light_green")

# Fazer as variaveis para por no objeto

# encontrar melhor set de pesos e biases
best_index = np.argmax(opt_accuracy)
# fazer variaveis baseado na melhor
best_accuracy = opt_accuracy[best_index]
best_loss = opt_loss[best_index]
best_weights = opt_w[best_index]
best_bias = opt_b[best_index]

# Fazer o objeto
object_final = {
    "Date": F.unixTimeToHumanReadable(time.time()),
    "Weights": list(best_weights),
    "Biases": list(best_bias),
    "Accuracy": best_loss,
    "Loss": best_accuracy,
    "Neurons_Positions": neural_network_layers_and_its_neurons
}

# por dados da melhor inteligencia artificial na base de dados
try:
    with open("D:\Python\Databases\Databases Weights And Biases\Final_Database_weights_and_biases_" + str(
            crypto_name) + ".json", "w") as file:
        json.dump(object_final, file, indent=2)
except Exception as e:
    print("An error occured:", e)
