import math as m
import numpy as np
import Functions as F
import pprint
import json
import time

crypto_name = "SOL"

#Isto é para a inteligência artificial, mas falta na mesma fazer um programa que faça trades por mim, e obviamente esse programa vai ter de ter pelo menos 3 funções: uma para calcular valor do portfólio pelo o que a neural network diz que vai subir, outro que pega numa API de um forum e diz se tem vários artigos a dizer que é scam e outra a ativar uma paragem de  emergência 

#datasets

#48 de preço(2 p/ hora), 24 de volume(1 p/ hora)/capitalizaçao de mercado, 24 de RSI(1 p/ hora), 12 de google trends(1 p/ 2 horas), 1 de market cap, 1 de market cap/TVL
inicial_inputs = []



#12 horas depois do último preço
#36 horas de dados, no total.
expected_outputs = []


with open("database_"+str(crypto_name)+"2_T.json", "r") as file:
    jr = json.load(file)
    for i in range(1, 201, 10):
        i = str(i)
        prices = jr[i]["Prices"]
        volume = jr[i]["Volume"]
        RSI = jr[i]["RSI"]
        Market_cap = jr[i]["Market_cap"]
        market_cap_TVL = jr[i]["MC/TVL"]
        outputs = jr[i]["Price_12_hours"]
        inputs_a = prices + volume + RSI + Market_cap + market_cap_TVL
        inicial_inputs.append(inputs_a)
        expected_outputs.append(outputs)

X = inicial_inputs[:]

Loss = 1

a3 = 0

def forward(inputs): #ReLu Activation a usar numpy
    outputs = np.maximum(0, inputs)
    return outputs

class Layer_Dense: #adaptado do curso do Youtube
    def __init__(self, n_neurons, weigths, biases):
        self.n_neurons = np.array(n_neurons)
        self.weigths = np.array(weigths)
        self.biases = np.array(biases)
    def forward(self, inputs):
        self.output = np.dot(self.weigths, inputs) + self.biases


input_layer_n_neurons = len(inicial_inputs[0])

a8 = False
a5 = 1
scale = 0.01
begginer_number = 1e-2
neural_network_layers_and_its_neurons = [input_layer_n_neurons, 100, 99, 50, 29, 1] #se tivesse mais de um output poria mais um layer


weigths_list_of_matrises_for_continuation = F.random_weights_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)
biases_matrix_continuation = F.random_bias_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)


'''
with open("Database_weights_and_biases_3"+str(crypto_name)+".json", "r") as file:
   load = json.load(file)
   weigths_list_of_matrises_for_continuation = load["Weights"]
   biases_matrix_continuation = load["Biases"]
   best_set_of_biases_yet = load["Biases"]
   best_set_of_weights_yet = load["Weights"]
   loss_b = load["Loss"]
'''


while Loss > 1e-4:
    loss = []
    accuracy = []
    a5 += 1
    for i in range(0, 10000):

        real_outputs = []

        for i in X:
            Hidden_Layer_1 = Layer_Dense(neural_network_layers_and_its_neurons[1], weigths_list_of_matrises_for_continuation[0], biases_matrix_continuation[0])
            Hidden_Layer_1.forward(i)

            output_layer_1 = forward(Hidden_Layer_1.output)

            Hidden_Layer_2 = Layer_Dense(neural_network_layers_and_its_neurons[2], weigths_list_of_matrises_for_continuation[1], biases_matrix_continuation[1])
            Hidden_Layer_2.forward(output_layer_1)

            output_layer_2 = forward(Hidden_Layer_2.output) 

            Hidden_Layer_3 = Layer_Dense(neural_network_layers_and_its_neurons[3], weigths_list_of_matrises_for_continuation[2], biases_matrix_continuation[2])
            Hidden_Layer_3.forward(output_layer_2)

            output_layer_3 = forward(Hidden_Layer_3.output)

            Hidden_Layer_4 = Layer_Dense(neural_network_layers_and_its_neurons[4], weigths_list_of_matrises_for_continuation[3], biases_matrix_continuation[3])
            Hidden_Layer_4.forward(output_layer_3)

            output_layer_4 = forward(Hidden_Layer_4.output)

            Output_layer = Layer_Dense(neural_network_layers_and_its_neurons[5], weigths_list_of_matrises_for_continuation[4], biases_matrix_continuation[4])
            Output_layer.forward(output_layer_4)

            output = Output_layer.output

            real_outputs.append(output)

        loss_a = F.accuracy_and_or_loss_in_one_output_NN(expected_outputs, real_outputs, 1)
        accuracy_a =  F.accuracy_and_or_loss_in_one_output_NN(expected_outputs, real_outputs, 2)

        accuracy.append(accuracy_a)
        loss.append(loss_a)

        a9 = True

        Loss = (sum(loss)/len(loss))

        if loss_a < loss_b:
            loss_b = loss_a
            best_set_of_weights_yet = weigths_list_of_matrises_for_continuation[:]
            best_set_of_biases_yet = biases_matrix_continuation[:]
            a8 = True
            a3 = 1

        weigths_list_of_matrises_for_continuation = F.matriz_muda_pesos(best_set_of_weights_yet, scale)
        biases_matrix_continuation = F.muda_biases(best_set_of_biases_yet, scale)

        print("Accuracy", accuracy_a, "Loss:", loss_a, sep = "\n")


        if a8:
            obj = {
                "Date": int(time.time()),
                "Weights": list(best_set_of_weights_yet),
                "Biases": list(best_set_of_biases_yet),
                "Accuracy:": accuracy_a,
                "Loss": loss_a,
                "Neurons_list":list(neural_network_layers_and_its_neurons)
            }
            try:
                with open("Database_weights_and_biases_3"+str(crypto_name)+".json", "w") as file:
                    json.dump(obj, file, indent=2)
                print("Data written to database.json successfully.")
            except Exception as e:
                print("An error occurred:", e)

        a8 = False