
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import layers
from tkinter.filedialog import askopenfilename
from PIL import Image
import numpy as np


#Variaveis 

height = 224
width = 224
batch = 32
epocas = 20
seed = 1313

path = askopenfilename()  #Escolhendo a imagem a ser predita


#Carrregando o modelo
model = load_model("D:/GitHub/OPCNN/resnet/Model.h5")

#Carregando os pesos
model.load_weights("D:/GitHub/OPCNN/resnet/ModelWeights.h5")

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

#Realizando o predict
pred = model.predict(input_arr)
print(pred.argmax())

#Plotando a imagem com o label predito
plt.imshow(image)
plt.title(labels[pred.argmax()])
plt.show()





