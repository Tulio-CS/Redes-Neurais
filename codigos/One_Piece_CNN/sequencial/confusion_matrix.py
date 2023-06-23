import seaborn as sn
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

path = "D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/Data/"

seed = 123
height = 224
width = 224
batch = 32

model = load_model("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/working/Model.h5")
model.load_weights("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/working/ModelWeights.h5")

pred_ds = train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    seed=seed,
    image_size=(height,width),
    batch_size= batch,
    label_mode="categorical",
    color_mode="rgb"
)

y_pred = model.predict(pred_ds)

y_pred_labels = [i.argmax() for i in y_pred]

pred_label = np.concatenate([y for x, y in pred_ds], axis=0)


y_test = (np.where(pred_label == 1)[1])

model.evaluate(pred_ds)
print(y_test,y_pred_labels)

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Truth')
plt.xlabel('Predicted')

plt.show()
