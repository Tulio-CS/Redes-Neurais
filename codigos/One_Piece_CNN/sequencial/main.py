

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers, callbacks

path = "D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/Data/"

height = 224
width = 224
batch = 32
epocas = 20
seed = 25

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

class_names = train_ds.class_names

print(class_names)

"""
plt.figure(figsize=(10,10))

for images,labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)      
        plt.imshow(images[i]/255)
        plt.title (class_names[labels[i]])
        plt.axis("off")
        break
    plt.show()
"""

model = Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.save("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/working/Model.h5")

checkpoint_filepath = 'D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/workingcp.ckpt'

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epocas,
    batch_size=batch,
    callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)

model.save_weights("D:/GitHub/Redes-Neurais/codigos/One_Piece_CNN/sequencial/working/ModelWeights.h5")


model.evaluate(train_ds)
model.evaluate(val_ds)

fig ,(ax1,ax2) = plt.subplots(2)

ax1.plot(history.history["loss"],label = "Loss")
ax1.plot(history.history["val_loss"],label = "Valid")
ax1.legend(loc = "upper right")

ax2.plot(history.history["accuracy"],label = "train")
ax2.plot(history.history["val_accuracy"],label = "Valid")
ax2.legend(loc = "upper right")

plt.show()