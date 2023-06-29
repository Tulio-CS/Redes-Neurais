

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
import keras.callbacks as tfc

#Variaveis

path = "D:/GitHub/OPCNN/10_classes/"     #Caminho com o dataset

height = 224          #Altura da imagem
width = 224           #Largura da imagem
batch = 32            #Tamanho do batch
epocas = 50           #Numero de epocas
seed = 13             #Seed aleatoria

#Criando os datasets

#Dataset para o treinamento
train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(height,width),
    batch_size= batch,
    label_mode="categorical",
    color_mode="rgb"
)

#Dataset para a validação
val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(height,width),
    batch_size= batch,
    label_mode="categorical",
    color_mode="rgb"
)

class_names = train_ds.class_names          #Nomes das classes

#Criando o modelo da rede
model = Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Salvando o modelo
model.save("D:/GitHub/OPCNN/sequencial/Model.h5")

#Criando o checkpoint, para salvar os melhores pesos
callback = tfc.ModelCheckpoint("D:/GitHub/OPCNN/sequencial/best.h5",save_best_only=True)

#Criando uma condicao para que a rede pare de treinar se nao houver melhoras, ajuda a evitar overfitting
early_stopping_callback = tfc.EarlyStopping(patience=5,restore_best_weights=True)  

#Treinando o modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epocas,
    batch_size=batch,
    callbacks=[early_stopping_callback, callback])

#Carregando os melhores pesos
model.load_weights("D:/GitHub/OPCNN/sequencial/best.h5")

#Salvando os pesos
model.save_weights("D:/GitHub/OPCNN/sequencial/ModelWeights.h5")


#Plotando o grafico de acuracia e loss
fig ,(ax1,ax2) = plt.subplots(2)

ax1.plot(history.history["loss"],label = "Loss")
ax1.plot(history.history["val_loss"],label = "Valid")
ax1.legend(loc = "upper right")

ax2.plot(history.history["accuracy"],label = "train")
ax2.plot(history.history["val_accuracy"],label = "Valid")
ax2.legend(loc = "upper right")

plt.show()