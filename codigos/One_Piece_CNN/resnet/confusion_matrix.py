import seaborn as sn
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Variaveis
data_path = "D:/GitHub/OPCNN//10_classes/"     #Caminho do dataset
model_path = "D:/GitHub/OPCNN/resnet/"         #Caminho do modelo
seed = 13
height = 224
width = 224
batch = 32


#Carregando o modelo
model = load_model(model_path+"Model.h5")

#Carregando os pesos
model.load_weights(model_path+"ModelWeights.h5")


#Criando o dataset
pred_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    image_size=(height,width),
    batch_size= batch,
    label_mode="int",
    color_mode="rgb",
    shuffle=False
)

#Realizando o predict no dataset
y_pred = model.predict(pred_ds)

model.evaluate(pred_ds)

#Criando a matriz de confusão
predictions = np.argmax(y_pred,axis=1)
truth = np.concatenate([y for x, y in pred_ds], axis=-1) 

cm = tf.math.confusion_matrix(labels=truth, predictions=predictions)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Truth')
plt.xlabel('Predicted')

plt.show()



