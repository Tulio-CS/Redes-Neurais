import seaborn as sn
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Variaveis
seed = 13
height = 224
width = 224
batch = 32

path = "D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN//test/"

#Carregando o modelo
model = load_model("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/resnet/working/Model.h5")

#Carregando os pesos
model.load_weights("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/resnet/working/ModelWeights.h5")


#Criando o dataset
pred_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    seed=seed,
    image_size=(height,width),
    batch_size= batch,
    label_mode="int",
    labels= "inferred",
    color_mode="rgb",
)

#Realizando o predict no dataset
y_pred = model.predict(pred_ds)
print(y_pred)

#Colhendo os valores preditos
y_pred_labels = [i.argmax() for i in y_pred]
print(y_pred_labels)

#Colhendo os valores que deveriam ser preditos
pred_label = np.concatenate([y for x, y in pred_ds], axis=0)
#y_test = (np.where(pred_label == 1))
print(pred_label)

#Avaliando o modelo
#model.evaluate(pred_ds)


#Criando a matriz de confusao
cm = tf.math.confusion_matrix(labels=pred_label, predictions=y_pred_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Truth')
plt.xlabel('Predicted')

plt.show()
