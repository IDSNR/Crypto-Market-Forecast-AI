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

def media_matrix(matrix):
    matri = []
    for i in matrix:
        matri.append(sum(i)/len(i))
    return matri

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

def listas_to_matrtix(lista_len_1st, lista_len_2nd):
    matrix_resultante = []
    for i in lista_len_1st:
        lista_a = []
        for j in lista_len_2nd:
            lista_a.append(i*j)
        matrix_resultante.append(lista_a)
    return matrix_resultante