import json
import time
import Functions as F

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

cryptos = ["ADA", "SOL", "WLD", "MATIC", "XRP", "ETH"]
list_er = []
number_of_instances = 3
lista_dia = ["02/03", "03/03", "04/03", "05/03"]
lista_percentagens = [-0.014, 0.036, 0.011, -0.029]

for a, i in enumerate(cryptos):
    list_er.append([])
    print("For:", i)
    for j in range(number_of_instances+1):
        print("Dia:", lista_dia[j])
        preco = float(input("Preço meu tropa:"))
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
        "Média": sum(lr) / len(lr)
    },
    "Percetagens crypto AI": {
        "Lista": lista_percentagens,
        "Soma": sum(lista_percentagens),
        "Média": sum(lista_percentagens) / len(lista_percentagens)
    },
    "Mercado": {
        "Lista": differences_between_days,
        "Soma": sum(differences_between_days),
        "Média": sum(differences_between_days) / len(differences_between_days)
    },
}

with open("Dados_Comparados_Com_Mercado.json", "r") as file:
    data = json.load(file)

data[F.unixTimeToHumanReadable(time.time())] = obj

with open("Dados_Comparados_Com_Mercado.json", "w") as file:
    json.dump(data, file, indent=2)



