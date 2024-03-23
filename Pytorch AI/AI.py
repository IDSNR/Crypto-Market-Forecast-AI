# Importing dependencies
import numpy as np
import torch
import json
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import adam
from torch.optim import sgd
import requests
import matplotlib.pyplot as plt
import random

def retorna_menor_index(lista):
    min = lista[0]
    index = 0
    for a, i in enumerate(lista):
        if i < min:
            index = a
            min = i
    return index

def Baralha_Listas(lista1, lista2, maximo_len=None):
    indexes = []
    for a, i in enumerate(lista1):
        indexes.append(a)
    if maximo_len is None:
        maximo_len = min(len(lista1), len(lista2))
    else:
        maximo_len = min(maximo_len, min(len(lista1), len(lista2)))
    lista1_baralhada = []
    lista2_baralhada = []
    lista_indexes = set()
    while len(lista1_baralhada) < maximo_len and len(lista2_baralhada) < maximo_len:
        random_index2 = random.randint(0, len(indexes)-1)
        random_index = indexes[random_index2]
        if random_index not in lista_indexes:  # Verificar se o índice já foi utilizado
            lista_indexes.add(random_index)
            lista1_baralhada.append(lista1[random_index])
            lista2_baralhada.append(lista2[random_index])
        del indexes[random_index2]
    return lista1_baralhada, lista2_baralhada


def test_train(X, y, test_size=0.2):
    X2_train = []
    y2_train = []
    X2_test = []
    y2_test = []
    le = len(X)
    len_per_train = round((1-test_size)*le)
    len_per_test = int(le - len_per_train)
    X_bar, Y_bar = Baralha_Listas(X, y)
    for i in range(len_per_train):
        y2_train.append(Y_bar[i])
        X2_train.append(X_bar[i])
    for j in range(len_per_test):
        X2_test.append(X_bar[j+len_per_train])
        y2_test.append(Y_bar[j+len_per_train])
    return X2_train, X2_test, y2_train, y2_test


new = False
training = True
accur = True
training_many = False
learning_rate = 0.001
iterations = 100000
epochs = 100

# Loading Data
crypto = "LTC"
try:
    with open("D:\Python\Databases\database_" + str(crypto) + "_T.json", "r") as file:
        jr = json.load(file)
except Exception as e:
    print("An error occured:", e)

# Initializing some variables
X = []
y_true = []


# Getting the data organized
for i in range(1, 490, 1):
  i = str(i)
  prices = jr[i]["Prices"]
  volume = jr[i]["Volume"]
  RSI = jr[i]["RSI"]
  Google_Trends_Data = jr[i]["Google Trends"]
  Market_cap = jr[i]["Market_cap"]
  market_cap_TVL = jr[i]["MC/TVL"]
  outputs = jr[i]["Price_12_hours"]
  inputs_a = prices + volume + RSI + Google_Trends_Data + Market_cap + market_cap_TVL
  if len(inputs_a) == 207:
    X.append(inputs_a)
    if outputs > 0.1:
        y_true.append([1, 0])
    else:
        y_true.append([0, 1])


def One_Hot_Encoded(y_true):
  y = []
  for i in y_true:
    y.append(np.argmax(i))
  return y

# [0, 1] --> Sell, [1, 0] --> Buy

X_train, X_test, y_train, y_test = test_train(X, y_true)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


#Creating Class with pytorch

class Model(nn.Module):
  #Input Layer (=length of inputs) --> Hidden Layer (x5) --> Output layer(Buy or sell)
  def __init__(self, in_features=207, h1=140, h2=100, h3=80, h4=50, h5=30, h6=16, h7=5, ol=2):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, h4)
    self.fc5 = nn.Linear(h4, h5)
    self.fc6 = nn.Linear(h5, h6)
    self.fc7 = nn.Linear(h6, h7)
    self.out = nn.Linear(h7, ol)

  def forward(self, X):
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = F.relu(self.fc3(X))
    X = F.relu(self.fc4(X))
    X = F.relu(self.fc5(X))
    X = F.relu(self.fc6(X))
    X = F.relu(self.fc7(X))
    X = self.out(X)
    #X = F.softmax(self.out(X))

    return X

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.parameters()

if training:
    using = 1
    loss_opt = []
    for i in range(iterations):
        y_pred = model.forward(X_train)

        loss = criterion(y_pred, y_train)
        #accuracy_m = accuracy(y_true, y_pred)

        loss_opt.append(loss.detach().numpy())

        if i % 100 == 0:
            print("Epoch:", i+(iterations*(j)), "Loss:", loss) #"Accuracy:", accuracy_m)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        using += 1

        if i % 5000 == 0:
            torch.save(model.state_dict(), "D:\Python\Databases\database_"+str(crypto)+"_T.json")

#Training
if training_many:
    using = 1
    loss_final = []
    for j in range(epochs):
        model = Model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.parameters()
        loss_opt = []
        for i in range(iterations):
            y_pred = model.forward(X_train)

            loss = criterion(y_pred, y_train)
            # accuracy_m = accuracy(y_true, y_pred)

            loss_opt.append(loss.detach().numpy())

            if i % 10 == 0:
                print("Epoch:", i + (iterations * (j)), "Loss:", loss)  # "Accuracy:", accuracy_m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            using += 1

            if i % 5000 == 0:
                torch.save(model.state_dict(), "D:\Python\Databases\database_"+str(crypto)+"_"+str(j)+"_T.json")

            if i == iterations-1:
                loss_final.append(loss.detach().numpy())
        for _ in range(100):
            print("New Model")
    print(retorna_menor_index(loss_final))

if accur:
    correct = 0
    with torch.no_grad():
      y_val = model.forward(X_test)
      loss = criterion(y_val, y_test)

    print("Loss:", loss)

    #Accuracy
    with torch.no_grad():
      for i, data in enumerate(X_test):
        y_eval = model.forward(data)
        if y_eval.argmax().item() == y_test[i].argmax():
          correct += 1

        #print(i, str(y_val.argmax()))
    for i, j in zip(y_val, y_test):
      print("Prediction:", i.argmax())
      print("Real:", j.argmax())



    print(correct)
    print(len(y_test))

    print(correct / len(y_test))


if new:
    new_data = torch.tensor()
    prediction = (model.forward(new_data)).argmax()
    if prediction == 0:
        print("Buy")
    if prediction == 1:
        print("Sell")
        print(prediction)
