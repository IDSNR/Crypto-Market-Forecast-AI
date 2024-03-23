import numpy as np
import math
import random
from pprint import pprint
import datetime

def true_len_matrix(matrix):
    soma = 0
    for i in matrix:
        for j in i:
            soma += 1
    return soma

def true_len_list_of_matrises(list_of_matrises):
    soma = 0
    for i in list_of_matrises:
        for j in i:
            for k in j:
                soma += 1
    return soma

def soma_lista(lista):
    soma = 0
    for i in lista:
        soma += i
    return soma

def media_lista(lista):
    soma_elementos = soma_lista(lista)
    media = soma_elementos / (len(lista))
    return media

def media_matriz(matrix):
    resultado = 0
    for i in matrix:
        resultado += media_lista(i)
    resultado_real = resultado / (len(matrix))
    return resultado_real

def multiplica_listas(lista1, lista2):
    resultado = 0
    for i, j in zip(lista1, lista2):
        resultado += i * j

def subtração_listas_elemento(lista1, elemento):
    lista_resultado = []
    for i in lista1:
        a = i - elemento
        lista_resultado.append(elemento)

def soma_listas(lista1, lista2):
    resultado = 0
    for i, j in zip(lista1, lista2):
        resultado += i + j

def soma_elementos_lista(lista1, lista2):
    lista_resolvida = []
    for i, j in zip(lista1, lista2):
        lista_resolvida.append(i + j)
    return lista_resolvida

def scale(lon):
    objeto_tipo_lista = lon[:]
    escala = []
    escala.append(len(objeto_tipo_lista))
    if type(objeto_tipo_lista[0]) != list and type(objeto_tipo_lista[0]) != tuple:
        return escala
    objeto_tipo_lista = objeto_tipo_lista[0]
    nested_escala = scale(objeto_tipo_lista)
    escala.extend(nested_escala)
    return escala

def real_len(lon, ajudante2=1, resultado2=0):
    objeto_tipo_lista = lon[:]
    if not isinstance(objeto_tipo_lista[0], (list, tuple)):
        return ajudante2 * len(objeto_tipo_lista) + resultado2
    ajudante = len(objeto_tipo_lista)
    objeto_tipo_lista = objeto_tipo_lista[0]
    return real_len(objeto_tipo_lista, ajudante, resultado2)

def all_layers_output(inputs, biases_matrix, weights_list_of_matrixes):
    n_neurons = true_len_matrix(biases_matrix)
    n_layers = n_neurons / len(biases_matrix)
    outputs = []
    for i, j in zip(weights_list_of_matrixes, biases_matrix):
        for k in i:
            output1 = multiplica_listas(inputs, k)
            output2 = soma_lista(output1, j)
            inputs = output2
            outputs.append(output2)
    assert n_layers == len(outputs)
    return outputs

def factorial(n):
    resultado = 1
    if n == 0:
        return 1
    else:
        for n in range(1, n + 1):
            resultado *= n
    return resultado

def returns_e():
    resultado = 0
    for i in range(0, 1000):
        resultado += (1 / factorial(i))
    return resultado

def rectified_linear(input, weight, bias):
    if input < 0:
        return (0 + bias)
    if input > 0:
        return (input * weight) + bias
    
def rectified_linear_1_neuron(inputs, weightsm, bias):
    resultado = 0
    for i, j in zip(inputs, weightsm):
        resultado += i * j
    if resultado <= 0:
        return bias
    elif resultado > 0:
        return (resultado + bias)
    
def forward_(inputs):
    a = np.maximum(0, inputs)
    return a

def correlação_entre_listas(lista1, lista2):
    soma_elementos_todas_as_listas = soma_listas(lista1, lista2)

def multiplica_listas(lista1, lista2):
    lista_resultante = []
    for i, j in zip(lista1, lista2):
        lista_resultante.append(i * j)
    return lista_resultante

def divide_listas(lista, n):
    lista_resultante = []
    for i in lista:
        lista_resultante.append(i/n)
    return lista_resultante

def softmax_activation(inputs):
    a = retorna_maior_indice_lista(inputs, 1)
    divided_values = divide_listas(inputs, a)
    exponentiated_values = []
    E = math.e
    for i in divided_values:
        exponentiated_values.append(E**i)  # Corrected line
    sum_all_exponentiated_values = sum(exponentiated_values)+1e-10
    percentage_values = []
    for i in exponentiated_values:
        percentage_values.append(i/sum_all_exponentiated_values)
    return percentage_values

def matrises_correct_confidencies_computer_output_multiple_neurons(y_pred, y_true):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1+1e-7)
    correct_confidencies = []
    for j, i in enumerate(y_true):
        correct_confidencies.append(y_pred_clipped[j][i])
    return correct_confidencies

def accuracy(y_true, y_confidence):
    a1 = 0
    assert len(y_true) == len(y_confidence)
    for i, j in zip(y_true, y_confidence):
        if i == j:
            a1 += 1
    return a1/len(y_true)

def index(lista, elemento):
    for i in range(0, lista):
        if lista[i] == elemento:
            return i

def maior_elemento_lista(lista):
    lista2 = sorted(lista, reverse=True)
    return lista2[0]

def index_maximum_elelment_list(lista):
    a = lista[0]
    for i in lista:
        if i > a:
            a = i
    return index(lista, a)

def accuracy_without_y_confidence_sorted_out(y_true, y_confidence_unsorted):
    y_confidence_sorted = []
    for i in y_confidence_unsorted:
        y_confidence_sorted.append(index_maximum_elelment_list(i))
    return accuracy(y_true, y_confidence_sorted)

def accuracy_and_or_loss_in_one_output_N(y_true, y_answers, returns = 0):
    assert len(y_true) == len(y_answers)
    soma = 0
    for i, j in zip(y_true, y_answers):
        error = abs(i - j)/i
        soma += error
    loss = soma/len(y_true)
    accuracy = 1 - loss
    if returns == 0:
        return loss, accuracy
    elif returns == 1:
        return loss
    elif returns == 2:
        return accuracy
    
def accuracy_and_or_loss_in_one_output_NNN(y_true, y_answers, returns = 0):
    assert len(y_true) == len(y_answers)
    lista_a1 = []
    for i, j in zip(y_true, y_answers):
        lista_a1.append(abs(i - j))
    lista_a2 = lista_a1[:]
    for i, j in enumerate(lista_a1):
        for k in y_true:
            if k == 0:
                k = 1e-15
            lista_a2[i] = j/k
    loss = (sum(lista_a2)/len(lista_a2))
    accuracy = 1 - loss
    lista_a1 = lista_a1
    lista_a2 = lista_a2
    if returns == 0:
        return loss, accuracy
    elif returns == 1:
        return loss
    elif returns == 2:
        return accuracy
    elif returns == 3:
        return lista_a1
    elif returns == 4:
        return lista_a2
    
def accuracy_and_or_loss_in_one_output_NN(y_true, y_answers, returns=0):
    assert len(y_true) == len(y_answers)

    y_true_np = np.array(y_true)
    y_answers_np = np.array(y_answers)

    absolute_errors = np.abs(y_true_np - y_answers_np)
    absolute_errors_percentage = absolute_errors / np.maximum(np.abs(y_true_np), 1e-15)

    loss = np.mean(absolute_errors)
    accuracy = 1 - loss

    if returns == 0:
        return loss, accuracy
    elif returns == 1:
        return loss
    elif returns == 2:
        return accuracy
    elif returns == 3:
        return absolute_errors.tolist()
    elif returns == 4:
        return absolute_errors_percentage.tolist()
        
def random_floats(original_float,n = 1, scale=0.01):
    random_numbers = []
    scale = scale * original_float
    for i in range(1, n + 1):
        perturbation = random.uniform(-scale, scale)
        random_numbers.append(original_float + perturbation)
    return random_numbers

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

def random_weights_for_Inicialization_and_continuation(n_neurons_list_as_array_per_layer, n=1, scale = 0.1):
    n_layers = len(n_neurons_list_as_array_per_layer)
    list_of_matrises_weights_all_layers = []
    for i in range(1, len(n_neurons_list_as_array_per_layer)):
        matrix_weights_layer = []
        for k in range(n_neurons_list_as_array_per_layer[i]):
            a1 = random_floats(n, n_neurons_list_as_array_per_layer[i - 1], scale)
            matrix_weights_layer.append(a1)
        list_of_matrises_weights_all_layers.append(matrix_weights_layer)
    return list_of_matrises_weights_all_layers

def random_bias_for_Inicialization_and_continuation(n_neurons_list_as_array_per_layer, n=1, scale = 0.1):
    matrises_bias = []
    for i in range(1, len(n_neurons_list_as_array_per_layer)):
        bias_list = random_floats(n, n_neurons_list_as_array_per_layer[i], scale)
        matrises_bias.append(bias_list)
    return matrises_bias

def perturb_float(original_float, scale=1):
    scale = scale * original_float
    perturbation = random.uniform(-scale, scale)
    return original_float + perturbation

def perturb_float_og(original_float, scale=1):
    perturbation = random.uniform(-scale, scale)
    return original_float + perturbation

def matriz_muda_pesos(weigths_list_of_matrises, scale = 0.1):
    for a, i in enumerate(weigths_list_of_matrises):
        for b, j in enumerate(i):
            for c, k in enumerate(j):
                weigths_list_of_matrises[a][b][c] = perturb_float(weigths_list_of_matrises[a][b][c], scale)
    return weigths_list_of_matrises

def muda_biases(bias_matrix, scale = 0.1):
    for a, i in enumerate(bias_matrix):
        for b, j in enumerate(i):
            bias_matrix[a][b] = perturb_float(bias_matrix[a][b], scale)
    return bias_matrix

def matriz_muda_pesos_og(weigths_list_of_matrises, scale = 0.1):
    for a, i in enumerate(weigths_list_of_matrises):
        for b, j in enumerate(i):
            for c, k in enumerate(j):
                weigths_list_of_matrises[a][b][c] = perturb_float_og(weigths_list_of_matrises[a][b][c], scale)
    return weigths_list_of_matrises

def muda_biases_og(bias_matrix, scale = 0.1):
    for a, i in enumerate(bias_matrix):
        for b, j in enumerate(i):
            bias_matrix[a][b] = perturb_float_og(bias_matrix[a][b], scale)
    return bias_matrix

def linear_search(lst, target):
    for i, element in enumerate(lst):
        if element == target:
            return i
    return False

def retorna_menor_numero_de_lista(lista, returns=0):
    if not lista:
        return None 
    min_value = lista[-1]
    index = len(lista) - 1
    for i in range(len(lista) - 2, -1, -1):
        if lista[i] < min_value: 
            min_value = lista[i]
            index = i
    if returns == 0:
        return min_value
    elif returns == 1:
        return index
    elif returns == 2:
        return min_value, index
    else:
        return None  
    
def matriz_lista_normal(matriz):
    lista_resultante = []
    for i in matriz:
        for j in i:
            lista_resultante.append(j)

def tira_n_elemento_lista(lista, n=15):
    lista_resultante = []
    for i in range(0, len(lista), n):
        lista_resultante.append(lista[i])
    return lista_resultante

def media_lista(lista):
    return (sum(lista)/len(lista))

def media_elemttos_lista_média_por_n(lista, n):
    lista_resultante = []
    a1 = 0
    a2 = n
    a = True
    for i in range(0, len(lista), n):
        if len(lista[a1:a2]) < n:
            a = False 
        if a:
            lista_resultante.append(media_lista(lista[a1:a2]))
        a1 += n
        a2 += n
        a = True
    return lista_resultante

def string_of_numbers_to_list(text: str, delimiter: str = '\n'):
    lines = [line.split(delimiter) for line in text.split('\n')]
    lines = [line for line in lines if line]
    numbers = [float(subline) for line in lines for subline in line]
    return numbers

def media_elemttos_lista_média_por_n_2(lista, n):
    lista_resultante = []
    a1 = 0
    a2 = len(lista)//n
    a3 = len(lista)//n
    a = True
    for i in range(0, len(lista), a3):
        if len(lista[a1:a2]) < n:
            a = False 
        if a:
            lista_resultante.append(media_lista(lista[a1:a2]))
        a1 += n
        a2 += n
        a = True
    return lista_resultante

def media_elementos_lista_media_por_n_final(lista, n):
    lista_resultante = []
    segment_size = len(lista) // n
    remainder = len(lista) % n

    a1 = 0
    for i in range(n):
        a2 = a1 + segment_size + (1 if i < remainder else 0)
        lista_resultante.append(sum(lista[a1:a2]) / (a2 - a1))
        a1 = a2

    return lista_resultante

def retorna_menor_indice_lista(lista, returns = 0):
    index = 0
    m_n = lista[0]
    for a, i in enumerate(lista):
        if i < m_n:
            index = a
            m_n = i
    if returns == 0:
        return index, m_n
    if returns == 1:
        return index
    if returns == 2:
        return m_n
        
def retorna_maior_indice_lista(lista):
    return np.argmax(lista)
    
def subtrai_lista(lista, n_a_subtrair):
    for i in range(len(lista)):
        lista[i] = lista[i] - n_a_subtrair
    return lista
    
def Softmax(lista_de_outputs):
    lista1 = lista_de_outputs[:]
    maiior_numero = retorna_maior_indice_lista(lista1)
    lista1 = subtrai_lista(lista1, maiior_numero)

    if all(x == 0 for x in lista_de_outputs):
        raise ValueError("All elements in the input list are zero.")

    for i in range(len(lista1)):
        if lista1[i] == 0:
            lista1[i] = 1e-10

    for i in range(len(lista1)):
        lista1[i] = math.log(lista1[i])

    soma_elementos = sum(lista1)

    for i in range(len(lista1)):
        lista1[i] = lista1[i]/soma_elementos
    
    return lista1


def From_Softmax_to_0_arrays(lista):
    maior_indice = retorna_maior_indice_lista(lista)
    lista_resultante = []
    for i in lista:
        lista_resultante.append(0)
    lista_resultante[maior_indice] = 1
    return lista_resultante

def Accuracy(y_true, y_given, retunrs = 0):
    soma = 0
    for i, j in zip(y_true, y_given):
        if i == j:
            soma += 1
    if retunrs == 0:        
        return soma / len(y_true)
    if retunrs == 1:
        return soma
    

def Accuracy_many_ansers(y_true2, y_given2, returns = 0):
    y_true = []
    y_given = []
    for i, j in zip(y_true2, y_given2):
        y_true.append(retorna_maior_indice_lista(y_true2))
        y_given.append(retorna_maior_indice_lista(y_given2))
    soma = 0
    for i, j in zip(y_true, y_given):
        if i == j:
            soma += 1
    if returns == 0:
        return soma/len(y_true)
    if returns == 1:
        return soma


def Loss(y_true, y_given):
    maior_indicdee_true = retorna_maior_indice_lista(y_true)
    indice_y_given = y_given[maior_indicdee_true]

    epsilon = 1e-15
    indice_y_given = np.clip(indice_y_given, epsilon, 1 - epsilon)

    Loss = -np.log(indice_y_given)
    return Loss

def Many_Losses(y_true, y_given, returns = 0):
    lista_losses_real = []
    for i, j in zip(y_true, y_given):
        lista_losses_real.append(Loss(i, j))
    if returns == 0:
        return lista_losses_real
    if returns == 1:
        return sum(lista_losses_real)/len(lista_losses_real)
    
def unixTimeToHumanReadable(seconds):
 
    # Save the time in Human
    # readable format
    ans = ""
 
    # Number of days in month
    # in normal year
    daysOfMonth = [31, 28, 31, 30, 31, 30,
                   31, 31, 30, 31, 30, 31]
 
    (currYear, daysTillNow, extraTime,
     extraDays, index, date, month, hours,
     minutes, secondss, flag) = (0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0)
 
    # Calculate total days unix time T
    daysTillNow = seconds // (24 * 60 * 60)
    extraTime = seconds % (24 * 60 * 60)
    currYear = 1970
 
    # Calculating current year
    while (daysTillNow >= 365):
        if (currYear % 400 == 0 or
            (currYear % 4 == 0 and
                currYear % 100 != 0)):
          if daysTillNow < 366:
            break
          daysTillNow -= 366
 
        else:
            daysTillNow -= 365
 
        currYear += 1
 
    # Updating extradays because it
    # will give days till previous day
    # and we have include current day
    extraDays = daysTillNow + 1
 
    if (currYear % 400 == 0 or
        (currYear % 4 == 0 and
            currYear % 100 != 0)):
        flag = 1
 
    # Calculating MONTH and DATE
    month = 0
    index = 0
 
    if (flag == 1):
        while (True):
 
            if (index == 1):
                if (extraDays - 29 < 0):
                    break
 
                month += 1
                extraDays -= 29
 
            else:
                if (extraDays - daysOfMonth[index] < 0):
                    break
 
                month += 1
                extraDays -= daysOfMonth[index]
 
            index += 1
 
    else:
        while (True):
            if (extraDays - daysOfMonth[index] < 0):
                break
 
            month += 1
            extraDays -= daysOfMonth[index]
            index += 1
 
    # Current Month
    if (extraDays > 0):
        month += 1
        date = extraDays
 
    else:
        if (month == 2 and flag == 1):
            date = 29
        else:
            date = daysOfMonth[month - 1]
 
    # Calculating HH:MM:YYYY
    hours = extraTime // 3600
    minutes = (extraTime % 3600) // 60
    secondss = (extraTime % 3600) % 60
 
    ans += str(int(date))
    ans += "/"
    ans += str(int(month))
    ans += "/"
    ans += str(int(currYear))
    ans += " "
    ans += str(int(hours))
    ans += ":"
    ans += str(int(minutes))
    ans += ":"
    ans += str(int(secondss))
 
    # Return the time
    return ans

import termcolor as color

def probability_of_an_outcome(probability):
    #a single outcome, a single probability
    #returns True if it happens and False if it doesn´t

    if probability > 1:
        probability /= 100

    random_v = random.uniform(0, 1)

    if random_v <= probability:
        return True
    
    else:
        return False
    
def diferença_de_precos(prices):
    precos_resultantes = [0]
    for i, j in zip(prices[:-1], prices[1:]):
        precos_resultantes.append(j - i)
    return precos_resultantes

def get_results_n_on_n(lista, n, par = False, PSB = True, ultimo = True):
    assert n <= len(lista)
    lista_resultante = []

    if PSB:
        n = len(lista)//n

    if not par:
        pop = 0
    if par:
        pop = 1
    
    for a in range (pop, len(lista), n):
        lista_resultante.append(lista[a])

    if PSB:
        if len(lista_resultante) > n and ultimo:
            del lista_resultante[-1]
        elif len(lista_resultante) > n and (not ultimo):
            del lista_resultante[0]
    
    return lista_resultante

def temperature(time, loss_deviation):
    E = math.e
    loss_deviation_usable = 2/(1+E**loss_deviation)

    if time <= 1000:
        a1 = time * 0.0009
        time = 1 - a1
    else:
        time = 100 / time
        
    temperature = loss_deviation_usable * time

    return temperature

def learning_rate_opt(time, batch_size, over_n = 0.2):
    return over_n/(1 + math.e ** ((time/(batch_size * 0.2))))

def gradient_of_loss(true_value, predicted_value):
    gradient = predicted_value - true_value
    return gradient

def gradient_of_loss_a(lista):
    pass