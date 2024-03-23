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

def get_results_from_AI(neural_network_layers_and_its_neurons, i, weigths, biases):

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

def get_results_from_AI_supports_many_different_neurons(neural_network_layers_and_its_neurons, input, weigths, biases):

    for i in range(len(neural_network_layers_and_its_neurons)):
        Hidden_Layer_1 = Layer_Dense(neural_network_layers_and_its_neurons[i + 1], weigths[i], biases[i])
        Hidden_Layer_1.forward(input)
        output_layer = forward(Hidden_Layer_1.output)
        input = output_layer

    return output_layer

crypto_name = "BTC"

inputs = []

expected_outputs = []

with open("database_"+str(crypto_name)+"_T.json", "r") as file:
        jr = json.load(file)
        for j in range(251, 500, 5):
            j = str(j)
            prices = jr[j]["Prices"]
            volume = jr[j]["Volume"]
            RSI = jr[j]["RSI"]
            Market_cap = jr[j]["Market_cap"]
            market_cap_TVL = jr[j]["MC/TVL"]
            i = prices + volume + RSI + Market_cap + market_cap_TVL
            expected_output_l = jr[j]["Price_12_hours"]
            expected_outputs .append(expected_output_l)
            inputs.append(i)
    

with open("Database_weights_and_biases_"+str(crypto_name)+".json", "r") as file:
    load = json.load(file)
    weigths = load["Weights"]
    biases = load["Biases"]
    neural_network_layers_and_its_neurons = load["Neurons_list"]
    
outputs = []
     
for ai in inputs:
    outputs.append(get_results_from_AI(neural_network_layers_and_its_neurons, ai, weigths, biases))
    
media_outputs = sum(outputs)/len(outputs)

media_exp_out = sum(expected_outputs)/len(expected_outputs)

loss = abs(media_exp_out - media_outputs)/media_exp_out

accuracy = 1 - loss

print("Loss:", loss, "Accuracy:", accuracy, sep= "\n")

