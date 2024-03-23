import json
import numpy as np

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

def get_results_from_AI(crypto_name):
    with open("database_"+str(crypto_name)+"_B.json", "r") as file:
        jr = json.load(file)
        j = str(j)
        prices = jr[j]["Prices"]
        volume = jr[j]["Volume"]
        RSI = jr[j]["RSI"]
        Market_cap = jr[j]["Market_cap"]
        market_cap_TVL = jr[j]["MC/TVL"]
        i = prices + volume + RSI + Market_cap + market_cap_TVL

    with open("Database_weights_and_biases_"+str(crypto_name)+".json", "r") as file:
        load = json.load(file)
        weigths = load["Weights"]
        biases = load["Biases"]
        neural_network_layers_and_its_neurons = load["Neurons_list"]



    Hidden_Layer_1 = Layer_Dense(neural_network_layers_and_its_neurons[1], weigths[0], biases[0])
    Hidden_Layer_1.forward(i)

    output_layer_1 = forward(Hidden_Layer_1.output)

    Hidden_Layer_2 = Layer_Dense(neural_network_layers_and_its_neurons[2], weigths[1], biases[1])
    Hidden_Layer_2.forward(output_layer_1)

    output_layer_2 = forward(Hidden_Layer_2.output) 

    Hidden_Layer_3 = Layer_Dense(neural_network_layers_and_its_neurons[3], weigths[2], biases[2])
    Hidden_Layer_3.forward(output_layer_2)

    output_layer_3 = forward(Hidden_Layer_3.output)

    Hidden_Layer_4 = Layer_Dense(neural_network_layers_and_its_neurons[4], weigths[3], biases[3])
    Hidden_Layer_4.forward(output_layer_3)

    output_layer_4 = forward(Hidden_Layer_4.output)

    Output_layer = Layer_Dense(neural_network_layers_and_its_neurons[5], weigths[4], biases[4])
    Output_layer.forward(output_layer_4)

    output = Output_layer.output

    return output


with open("Database_for_executing.json", "r") as file:
    load = json.load(file)
    

#0 para só uma crypto, 1 para uma lista de cryptos    
list_or_not_list = 0


if list_or_not_list == 0:
    list_or_not_list = False
elif list_or_not_list == 1:
    list_or_not_list = True

crypto_list = ["SOL", "ADA"]

crypto_name = "SOL"

iteration_number = 0


if not list_or_not_list:
    output, expected = get_results_from_AI(str(crypto_name))

    print(output, expected, (abs(output-expected)/expected), sep="\n\n")


    j = {
        "Predicted Price": output
    }

    load[str(crypto_name)] = j

    with open("Database_for_executing.json", "w") as file:
        json.dump(load, file, indent=2)
        
if list_or_not_list:
    for crypt in crypto_list:
        output, expected = get_results_from_AI(str(crypt))

        print(output, expected, (abs(output-expected)/expected), sep="\n\n")


        j = {
            "Predicted Price": output
        }

        load[str(crypt)] = j
        
    with open("Database_for_executing.json", "w") as file:
        json.dump(load, file, indent=2)
        
def get_data_failure(crypto, crypto_list = False, last_number = 10000):
    try:
        with open("Database_for_executing.json", "r") as file:
            load = json.load(file)
    except Exception as e:
        print("An error happened:", e)
    
    data = []

    if not crypto_list:
        for i in range(1, last_number):
            try:
                data.append(load[str(crypto)+str(i)])
            except Exception as e:
                print("Exception:", e)

    if crypto_list:
        for crypto_ in crypto_list:
            data_ = []
            for i in range(1, last_number):
                data_.append(load[str(crypto_)+str(i)])
            data.append(data_)
            
    return data
    


def many_losses(lista_percentagem_e_ou_diferença, preco_crypto = 0.01, Diferenca = False, get_difference = False):
    soma = 0
    soma_ = 0
    soam = 0

    return_var = False

    for i in lista_percentagem_e_ou_diferença:
        soma += i

    if  soma <= 0:
        return_var = True
        
    if get_difference and (not Diferenca):
        for i in lista_percentagem_e_ou_diferença:
            soma_ += i * preco_crypto
        return return_var, soma_
    
    if get_difference and Diferenca:
        for i in lista_percentagem_e_ou_diferença:
            soam += i
        return return_var, soam
    
    return return_var
    