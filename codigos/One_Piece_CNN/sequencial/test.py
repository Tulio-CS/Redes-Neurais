
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

model = load_model("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/working/Model.h5")
model.load_weights("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/ModelWeights.h5")

"""   
image = Image.open(path)
image = image.resize((224,224))
image = numpy.array(image)
"""

image = tf.keras.utils.load_img(
    path,
    target_size=(224,224)
)

input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

labels = {0:"Ace",
          1:"Akainu",
          2:"Brook",
          3:"Chopper",
          4:"Crocodile",
          5:"Franky",
          6:"Jinbei",
          7:"Kurohige",
          8:"Law",
          9:"Luffy",
          10:"Mihawk",
          11:"Nami",
          12:"Rayleigh",
          13:"Robin",
          14:"Sanji",
          15:"Shanks",
          16:"Usopp",
          17:"Zoro"
          }

pred = model.predict(input_arr)
print(pred.argmax())
plt.imshow(image)
plt.title(labels[pred.argmax()])
plt.show()





