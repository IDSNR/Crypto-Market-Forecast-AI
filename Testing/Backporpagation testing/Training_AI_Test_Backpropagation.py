import math as m
import numpy as np
import Functions as F
import json
import time
import termcolor as color


crypto_name = "BTC"

#Isto é para a inteligência artificial, mas falta na mesma fazer um programa que faça trades por mim, e obviamente esse programa vai ter de ter pelo menos 3 funções: uma para calcular valor do portfólio pelo o que a neural network diz que vai subir, outro que pega numa API de um forum e diz se tem vários artigos a dizer que é scam e outra a ativar uma paragem de  emergência 

#datasets

#48 de preço(2 p/ hora), 24 de volume(1 p/ hora)/capitalizaçao de mercado, 24 de RSI(1 p/ hora), 12 de google trends(1 p/ 2 horas), 1 de market cap, 1 de market cap/TVL
inicial_inputs = []



#12 horas depois do último preço
#36 horas de dados, no total.
expected_outputs = []


with open("database_"+str(crypto_name)+"_T.json", "r") as file:
    jr = json.load(file)
    for i in range(1, 501, 2):
        i = str(i)
        prices = jr[i]["Prices"]
        volume = jr[i]["Volume"]
        RSI = jr[i]["RSI"]
        Market_cap = jr[i]["Market_cap"]
        #market_cap_TVL = jr[i]["MC/TVL"]
        outputs = jr[i]["Price_12_hours"]
        inputs_a = prices + volume + RSI + Market_cap #+ market_cap_TVL
        inicial_inputs.append(inputs_a)
        expected_outputs.append(outputs)

X = inicial_inputs[:]

Loss = 1

a3 = 1

def forward(inputs): #ReLu Activation a usar numpy
    #outputs = np.maximum(0, inputs)
    outputs = inputs[:]
    return outputs

class Layer_Dense: #adaptado do curso do Youtube
    def __init__(self, n_neurons, weigths, biases):
        self.n_neurons = np.array(n_neurons)
        self.weigths = np.array(weigths)
        self.biases = np.array(biases)
    def forward(self, inputs):
        self.inputs = np.array(inputs)
        self.output = np.dot(self.weigths, inputs) + self.biases
        self.output = np.array(self.output)
    def backpropagation(self, gradient, learning_rate = 0.01):
        gradient = np.array(gradient).flatten()
        print(self.inputs.T.shape)
        print(gradient.shape)

        weights_gradient = np.dot(gradient, self.output.T) / len(self.output)
        biases_gradient = np.sum(gradient) / len(self.output)

        self.weigths -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        self.weigths = np.array(self.weigths)

        return np.dot(self.output.T, gradient)


input_layer_n_neurons = len(inicial_inputs[0])



learning_rate = 0.01
a8 = False
a5 = 1
scale = 10
a12 = 0
Epoch_number = None
Epoch_or_not = False
begginer_number = 1e-5
Tottaly_Random = False
a15 = True
Batch_size = 10000
neural_network_layers_and_its_neurons = [input_layer_n_neurons, 10, 8, 6, 3, 1] #se tivesse mais de um output poria mais um layer


weigths_list_of_matrises_for_continuation = F.random_weights_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)
biases_matrix_continuation = F.random_bias_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)


'''
with open("Database_weights_and_biases_Test_"+str(iteration)+"_"+str(crypto_name)+".json", "r") as file:
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
    a15 = (a12 >= Epoch_number)
    

lists_opt_weights = []
lists_opt_biases = []
loss_opt = []
accuracy_opt = []

while Loss > 1e-4 and a15:
    
    if Epoch_or_not:
        a15 = (a12 >= Epoch_number)
    if a12 > 0:
        loss_opt.append(loss_b)
        accuracy_opt.append(accuracy_b)
    
    if Tottaly_Random:
        weigths_list_of_matrises_for_continuation = F.random_weights_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)
        biases_matrix_continuation = F.random_bias_for_Inicialization_and_continuation(neural_network_layers_and_its_neurons, begginer_number, scale)
    
    elif a12 > 0 and (not Tottaly_Random):
        weigths_list_of_matrises_for_continuation = F.matriz_muda_pesos_og(best_set_of_weights_yet, scale)
        biases_matrix_continuation = F.muda_biases_og(neural_network_layers_and_its_neurons, scale)
    
    neural_network_layers = [
    Layer_Dense(neural_network_layers_and_its_neurons[1], weigths_list_of_matrises_for_continuation[0], biases_matrix_continuation[0]),
    Layer_Dense(neural_network_layers_and_its_neurons[2], weigths_list_of_matrises_for_continuation[1], biases_matrix_continuation[1]),
    Layer_Dense(neural_network_layers_and_its_neurons[3], weigths_list_of_matrises_for_continuation[2], biases_matrix_continuation[2]),
    Layer_Dense(neural_network_layers_and_its_neurons[4], weigths_list_of_matrises_for_continuation[3], biases_matrix_continuation[3]),
    Layer_Dense(neural_network_layers_and_its_neurons[5], weigths_list_of_matrises_for_continuation[4], biases_matrix_continuation[4])
    ]
    
    
    a12 += 1
    
    a8 = True
    
    opt_b = []
    opt_w = []
    
    for i in neural_network_layers:
        opt_w.append(i.weigths)
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
                if a != (len(neural_network_layers)-1):
                    current_input = forward(layer.output)
                if a == (len(neural_network_layers)-1):
                    current_input = layer.output
                    current_input.tolist()
            
            current_input = current_input[0]

            real_outputs.append(current_input)
            
            loss_gradient = F.gradient_of_loss(expected_outputs[a21], current_input)

            current_gradient = loss_gradient
            for layer in reversed(neural_network_layers):
                current_gradient = layer.backpropagation(current_gradient, learning_rate)

        loss_a, accuracy_a = F.accuracy_and_or_loss_in_one_output_NN(expected_outputs, real_outputs, 0)

        accuracy.append(accuracy_a)
        loss.append(loss_a)
        
        if a12 == 1:
            loss_b = loss_a + 1e10

        Loss = (sum(loss)/len(loss))

        if loss_a < loss_b:
            opt_w2 = []
            opt_b2 = []
            
            loss_b = loss_a
            accuracy_b = accuracy_a
            
            color.cprint("\nYay", "green"), color.cprint("\nYay", "green"), color.cprint("\nYay", "green"), color.cprint("\nNovo set de biases e pesos encontrado!", "green"), color.cprint("\nAccuracy:", "green"), color.cprint(accuracy_a, "green"), color.cprint("\nLoss:", "green"), color.cprint(loss_a, "green")
            
            for i in neural_network_layers:
                opt_w2.append(i.weigths)
                opt_b2.append(i.biases)
                
            best_set_of_weights_yet = opt_w2[:]
            best_set_of_biases_yet = opt_b2[:]
            

        print("Accuracy", accuracy_a, "Loss:", loss_a, sep = "\n")


        if a8:
            obj = {
                "Date": int(time.time()),
                "Weights": list(best_set_of_weights_yet),
                "Biases": list(best_set_of_biases_yet),
                "Accuracy:": list(accuracy_a),
                "Loss": list(loss_b),
                "Neurons_list" : neural_network_layers_and_its_neurons
            }
            try:
                with open("Database_weights_and_biases_5"+str(crypto_name)+".json", "w") as file:
                    json.dump(obj, file, indent=2)
                print("Data written to database.json successfully.")
            except Exception as e:
                print("An error occurred:", e)
            
        a8 = False
        
        learning_rate = F.learning_rate_opt(a12, Batch_size)
        
        
        
a = F.retorna_menor_indice_lista(loss_opt, 1)

best_set_of_biases_yet = lists_opt_biases[a]
best_set_of_weights_yet = lists_opt_weights[a]
loss_b = loss_opt[a]
accuracy_b = accuracy_opt[a]

        
obj = {
"Date": int(time.time()),
"Weights": list(best_set_of_weights_yet),
"Biases": list(best_set_of_biases_yet),
"Accuracy:": list(accuracy_a),
"Loss": list(loss_b),
"Neurons_list" : neural_network_layers_and_its_neurons
}


try:
    with open("Database_weights_and_biases_4"+str(crypto_name)+".json", "w") as file:
        json.dump(obj, file, indent=2)
        print("Data written to database.json successfully.")
except Exception as e:
    print("An error occurred:", e)