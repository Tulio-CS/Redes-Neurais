
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import layers
from tkinter.filedialog import askopenfilename
from PIL import Image
import numpy as np
"""D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/archive/Data/Data/Nami/1.png"""
path = askopenfilename()


height = 224
width = 224
batch = 32
epocas = 20
seed = 13

#Carregando o modelo
model = load_model("E:/OPCNN/sequencial/Model.h5")

#Carregando os pesos do modelo
model.load_weights("E:/OPCNN/sequencial/ModelWeights.h5")

#Carregando a imagem
image = tf.keras.utils.load_img(
    path,
    target_size=(224,224)
)

#Convertendo a imagem em uma matriz
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

labels = {0:"Brook",
          1:"Chopper",
          2:"Franky",
          3:"Jinbei",
          4:"Luffy",
          5:"Nami",
          6:"Robin",
          7:"Sanji",
          8:"Usopp",
          9:"Zoro"
          }

#Predict e plotagem da imagem
pred = model.predict(input_arr)
print(pred)
#print(pred.argmax())
plt.imshow(image)
plt.title(labels[pred.argmax()])
plt.show()





