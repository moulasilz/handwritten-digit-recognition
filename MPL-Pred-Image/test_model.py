import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as img

model = pickle.load(open('model.pkl', 'rb'))
listdir = os.listdir(os.path.join(os.getcwd(), "uploads"))
image_arrays = np.array([[[0 for i in range(28)] for i in range(28)] for i in range(len(listdir))])
i = 0
for filename in listdir:
    image = img.open(os.path.join(os.getcwd()+ "/uploads", filename))
    image_arrays[i] = np.array(image)
    i += 1
    

image_arrays = image_arrays.reshape((len(image_arrays), -1))
predictions = model.predict(image_arrays)

def plot_multi_pred(images, pred, i = 0, nplots = None):
    if nplots == None:
        nplots = 16
    fig = plt.figure(figsize=(10,10))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(images[i+j].reshape(28, 28), cmap='binary')
        plt.title(pred[i+j])
        plt.axis('off')
    plt.show()
    
plot_multi_pred(image_arrays, predictions, nplots= len(image_arrays))
