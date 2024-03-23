import json
import sys
import Functions as F
import time
from pprint import pprint

def media_matrix_index(matrix):
    lr = []
    for i in range(len(matrix[0])):  # Loop over columns
        soma = 0
        for j in range(len(matrix)):  # Loop over rows
            soma += matrix[j][i]  # Access elements correctly
        lr.append(soma / len(matrix))  # Calculate average for each column
    return lr

def difference_percentage(lista):
    differences = []
    for i in range(len(lista)):
        if i != 0 and lista[i-1] != 0:
            differences.append((lista[i] - lista[i-1]) / lista[i-1])
    return differences

def subtrai_lista(lista, elemento):
    lr = []
    for i in lista:
        lr.append(i - elemento)
    return lr

def subtrai_listas(lista1, lista2):
    lr = []
    for i, j in zip(lista1, lista2):
        lr.append(i - j)
    return lr

def put_to_database(file_name:str, obj:dict, name=""):
    with open(file_name, "r") as file:
        data = json.load(file)
    if name == "":
        name = F.get_last_it(data)
    data[name] = obj
    with open(file_name, "w") as file:
        json.dump(data, file, indent=2)

cryptos = ["ADA", "SOL", "WLD", "MATIC", "XRP", "ETH"]
list_er = [[0.717658, 0.741776, 0.727649, 0.771212, 0.723533, 0.746853, 0.723946, 0.743705],
           [129.96, 129.88, 130.31, 133.28, 130.13, 146.86, 148.35, 145.65],
           [7.98, 7.7, 7.89, 7.39, 7.09, 7.13, 7.61, 10.76],
           [1.02, 1.09, 1.09, 1.15, 1.09, 1.17, 1.13, 1.13],
           [0.601021, 0.643575, 0.626699, 0.649938, 0.60993, 0.624719, 0.620466, 0.623245],
           [3433.11, 3421.89, 3487.19, 3633.77, 3852.31, 3861.67, 3930.32, 3897.68]]
number_of_instances = 3
lista_dia = ["02/03", "03/03", "04/03", "05/03", "06/03 - 18:00", "07/03 - 18:00", "08/03 - 18:45", "09/03 - 18:50"]
lista_percentagens = [-0.014, 0.036, 0.011, -0.029, 0.0155, 0.00015867, 0.125099]

how_many_missing = len(lista_dia) - len(list_er[0])

for a, i in enumerate(cryptos):
    print(i)
    for j in lista_dia[(len(lista_dia)-how_many_missing):]:
        print(j)
        preco = float(input("Preço:"))
        list_er[a].append(preco)

super = []
for prices in list_er:
    super.append(difference_percentage(prices))
differences_between_days = media_matrix_index(super)

lr = subtrai_listas(lista_percentagens, differences_between_days)

print("Lista:", lr, "Total:", sum(lr), "Média:", sum(lr)/len(lr), "Percentagens crypto:", lista_percentagens, "Mercado:", differences_between_days, "Média Crypto:", sum(lista_percentagens)/len(lista_percentagens), "Média Mercado:", sum(differences_between_days)/len(differences_between_days), sep="\n")

obj = {
    "Time": time.time(),
    "Percetagens crypto AI - Mercado": {
        "Lista": lr,
        "Soma": sum(lr),
        "Média": sum(lr)/len(lr)
    },
    "Percetagens crypto AI": {
        "Lista": lista_percentagens,
        "Soma": sum(lista_percentagens),
        "Média": sum(lista_percentagens)/len(lista_percentagens)
    },
    "Mercado": {
        "Lista": differences_between_days,
        "Soma": sum(differences_between_days),
        "Média": sum(differences_between_days)/len(differences_between_days)
    },
}

pprint(list_er)

with open("Dados_Comparados_Com_Mercado.json", "r") as file:
    data = json.load(file)

data[F.unixTimeToHumanReadable(time.time())] = obj

with open("Dados_Comparados_Com_Mercado.json", "w") as file:
    json.dump(data, file, indent=2)







