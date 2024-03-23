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

    outputs = []

    with open("database_"+str(crypto_name)+"_T.json", "r") as file:
        X = []
        expected_output = []
        jr = json.load(file)
        for j in range(170, 180):
            j = str(j)
            prices = jr[j]["Prices"]
            volume = jr[j]["Volume"]
            RSI = jr[j]["RSI"]
            Market_cap = jr[j]["Market_cap"]
            market_cap_TVL = jr[j]["MC/TVL"]
            i = prices + volume + RSI + Market_cap + market_cap_TVL
            expected_output.append(jr[j]["Price_12_hours"])
            X.append(i)

    with open("Database_weights_and_biases_3"+str(crypto_name)+".json", "r") as file:
        load = json.load(file)
        weigths = load["Weights"]
        biases = load["Biases"]
    
    neural_network_layers_and_its_neurons = [203, 100, 99, 50, 29, 1]   #load["Neurons_list"]


    for i in X:
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
        outputs.append(output)

    return outputs, expected_output

'''
with open("Database_for_executing.json", "r") as file:
    load = json.load(file)
'''

crypto_name = "XRP"
iteration_number = 0

output, expected = get_results_from_AI(str(crypto_name))

for i, j in zip(output, expected):
    print(i, j, (abs(i-j)/j))



#print(output, expected, (abs(output-expected)/expected), sep="\n\n")


'''j = {
    "P Price": output
}

load[str(crypto_name)+str(iteration_number)] = j

with open("Database_for_executing.json", "w") as file:
    json.dump(load, file, indent=2)'''