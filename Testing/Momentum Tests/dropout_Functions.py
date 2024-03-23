import random

def delete_n_random_elements_from_list(list: list, n_delete: int):
    a1 = len(list)
    list_r = list[:]
    lista_indices_deletados = []
    while (a1 - n_delete) < len(list_r):

        random_index = random.randint(0, len(list_r) - 1)
        lista_indices_deletados.append(list.index(list_r[random_index]))
        del list_r[random_index]

    return list_r, lista_indices_deletados

def delete_based_on_array_of_indixes(list_or_matrix, indexes_matrix):
    resultado = list_or_matrix[:]
    for i in indexes_matrix:
        del resultado[resultado.index(list_or_matrix[i])]
    return resultado

def delete_matrix_2nd_dim_by_list_of_indexes(matrix, index_array):
    matrix_r = []
    for j in matrix:
        matrix_r.append(delete_based_on_array_of_indixes(j, index_array))
    return matrix_r


