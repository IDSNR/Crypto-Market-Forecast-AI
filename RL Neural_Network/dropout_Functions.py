import random
import time

def delete_n_random_elements_from_list(list: list, n_delete: int):
    a2 = 0
    a1 = len(list)
    list_r = list[:]
    lista_indices_deletados = []
    while (a1 - n_delete) < len(list_r):

        random_index = random.randint(0, len(list_r) - 1)
        lista_indices_deletados.append(list.index(list_r[random_index]))
        del list_r[random_index]

    return list_r, lista_indices_deletados

def delete_list_based_on_elements(list, list_took_out):
    lista_resultante = []
    for a, i in enumerate(list):
        if not a in list_took_out:
            lista_resultante.append(i)
    return lista_resultante

def delete_based_on_array_of_indixes(list_or_matrix, indexes_matrix):
    if indexes_matrix:
        list_or_matrix = list_or_matrix.tolist()
        resultado = list_or_matrix[:]
        for i in indexes_matrix:
            del resultado[resultado.index(list_or_matrix[i])]
        return resultado
    return list_or_matrix

def delete_matrix_2nd_dim_by_list_of_indexes(matrix, index_array):
    matrix_r = []
    for j in matrix:
        matrix_r.append(delete_based_on_array_of_indixes(j, index_array))
    return matrix_r


def delete_n_rnd_elem_fr_lsss(list1: list, list2: list, n: int):
    a2 = 0
    a1 = len(list1)
    list_r1 = list1[:]
    lista_r2 = list2[:]
    lista_indices_deletados = []
    while (a1 - n) < len(list_r1):

        random_index = random.randint(0, len(list_r1) - 1)
        lista_indices_deletados.append(list1.index(list_r1[random_index]))
        del list_r1[random_index]
        del lista_r2[random_index]

    return list_r1, lista_r2, lista_indices_deletados


def delete_n_rnd_elem_fr_lsss2(list1: list, list2: list, n: int):
    list_r1 = list1[:]
    lista_r2 = list2[:]
    lista_indices_deletados = set()  # Using a set to store indices to delete
    list_length = len(list_r1)

    # Check if n is greater than the length of the list
    if n >= list_length:
        return [], [], []

    # Generate random indices to delete
    while len(lista_indices_deletados) < n:
        random_index = random.randint(0, list_length - 1)
        lista_indices_deletados.add(random_index)

    # Create new lists without the deleted elements
    list_r1 = [list_r1[i] for i in range(list_length-1) if i not in lista_indices_deletados]
    lista_r2 = [lista_r2[i] for i in range(list_length-1) if i not in lista_indices_deletados]

    return list_r1, lista_r2, list(lista_indices_deletados)
