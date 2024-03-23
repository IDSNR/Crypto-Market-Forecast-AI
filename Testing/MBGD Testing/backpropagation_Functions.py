import random

import numpy as np

def troca_elementos_matrix_soma(matrix_weights, gradient, position, how_many = 0):
    matrix_resultante = matrix_weights[:]
    if how_many == 0:
        how_many = len(matrix_weights)
    for i in range(0, how_many):
        matrix_resultante[i][position] += gradient
    return matrix_resultante


def troca_elementos_matrix_subtraçao(matrix_weights, gradient, position, how_many = 0):
    matrix_resultante = matrix_weights[:]
    if how_many == 0:
        how_many = len(matrix_weights)
    for i in range(0, how_many):
        matrix_resultante[i][position] -= gradient
    return matrix_resultante

def troca_elementos_matrix_divisao(matrix_weights, gradient, position, how_many = 0):
    matrix_resultante = matrix_weights[:]
    if how_many == 0:
        how_many = len(matrix_weights)
    for i in range(0, how_many):
        matrix_resultante[i][position] /= gradient
    return matrix_resultante

def troca_elementos_matrix_multiplicaçao(matrix_weights, gradient, position, how_many = 0):
    matrix_resultante = matrix_weights[:]
    if how_many == 0:
        how_many = len(matrix_weights)
    for i in range(0, how_many):
        matrix_resultante[i][position] *= gradient
    return matrix_resultante

def clipa_sem_a_min(lista, valor_máximo_ou_minimo, Máximio = True):
    if Máximio:
        return np.clip(lista, a_min=(lista[np.argmin(lista)]), a_max=valor_máximo_ou_minimo)
    if not Máximio:
        return np.clip(lista, a_max=(lista[np.argmax(lista)]), a_min=valor_máximo_ou_minimo)
def media_matrix(matrix):
    matri = []
    for i in matrix:
        matri.append(sum(i)/len(i))
    return matri

def media_por_indice_matrix(matrix):
    lista_resultante = []
    for i in range(len(matrix[0])):
        soma = 0
        for j in matrix:
            soma += j[i]
        lista_resultante.append(soma/len(matrix))
    return lista_resultante

def media_matrix_lista_posiçao(matrix, posicao):
    soma = 0
    for i in matrix:
        soma += i[posicao]
    return soma / len(matrix)

def soma_matrrix_em_lista(matrix):
    lista_resultante = []
    for i in matrix:
        lista_resultante.append(sum(i))
    return lista_resultante

def media_lista(lista):
    return sum(lista)/len(lista)

def probabilidade(prob):
    pro = random.uniform(0, 1)
    if prob >= pro:
        return True
    return False

def get_zeros_and_ones_list(len_list, prob_1):
    if not isinstance(len_list, int):
        len_list = len(len_list)
    lista_r = []
    for i in range(len_list):
        if probabilidade(prob_1):
            lista_r.append(1)
        else:
            lista_r.append(0)
    return lista_r

def zerosd_ones_to_matrix_zeros_ones(shape_matrix, ones_zeros_list):
    if not isinstance(shape_matrix, tuple):
        shape_matrix = (np.array(shape_matrix).shape)
    assert shape_matrix[0] == len(ones_zeros_list)
    matrix_resultante = []
    for i in range(shape_matrix[0]):
        lista_a = []
        for j in range(shape_matrix[1]):
            lista_a.append(ones_zeros_list[i])
        matrix_resultante.append(lista_a)
    return matrix_resultante

def multiplica_matrises(matrix, n):
    matrix_r = []
    for i in matrix:
        lista_a = []
        for j in i:
            lista_a.append(j * n)
        matrix_r.append(lista_a)
    return matrix_r

def listas_to_matrtix(lista_len_1st, lista_len_2nd):
    matrix_resultante = []
    for i in lista_len_1st:
        lista_a = []
        for j in lista_len_2nd:
            lista_a.append(i*j)
        matrix_resultante.append(lista_a)
    return matrix_resultante
