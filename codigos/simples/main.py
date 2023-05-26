
#Importar as bibliotecas
from tensorflow import random
from keras import layers, models, optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score

path = "teste00.xlsx"

DataFrame = pd.read_excel(path)


X_train = DataFrame.iloc[0:16,0:2].astype(float)
y_train = DataFrame.iloc[0:16,2].astype(float)
X_valid = DataFrame.iloc[16:25,0:2].astype(float)
y_valid = DataFrame.iloc[16:25,2].astype(float)
X_test = DataFrame.iloc[25:41,0:2].astype(float)
y_test = DataFrame.iloc[25:41,2].astype(float)

seed = 13 #galo doido/PT
random.set_seed(seed)

#Criando o modelo
model = models.Sequential()

model.add(layers.Dense(10, activation='relu', input_dim=2))

model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='linear'))

# Otimizador
model.compile(loss='mse', optimizer='Adam')

#Resumo do Modelo
model.summary()

#Treinamento
history = model.fit(X_train, y_train, epochs=50, batch_size=1, validation_data=(X_valid, y_valid), shuffle=True, verbose=1)

y_pred = model.predict(X_test)

metrics = {"MAE":mean_absolute_error(y_test,y_pred),"R2 SCORE":r2_score(y_test,y_pred)}

for key,value in metrics.items():
    print("{} = {}".format(key,value))


fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.plot(history.history["loss"],label = "Loss")
ax1.plot(history.history["val_loss"],label = "Valid")
ax1.legend(loc = "upper right")

ax2 = fig.add_subplot(122)
ax2.scatter(y_pred,y_test, color = "g")
ax2.plot(y_test.values,y_test.values,"r")
plt.show()