import matplotlib.pyplot as plt
import tensorflow as tf

path = "D:/GitHub/OPCNN/t/"
height = 224
width = 224

ds = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(height,width),
    label_mode="int",
    labels="inferred",
    color_mode="rgb"
)

class_names = ds.class_names

plt.figure(figsize=(10,10))

for images,labels in ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)      
        plt.imshow(images[i]/255)
        plt.title (class_names[labels[i]])
        print(labels[i])
        plt.axis("off")
    plt.show()