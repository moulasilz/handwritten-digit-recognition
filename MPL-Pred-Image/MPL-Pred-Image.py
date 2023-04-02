

import matplotlib.pyplot as plt 
import numpy as np

from sklearn import datasets
digits = datasets.load_digits()

dir(digits)

print(type(digits.images))
print(type(digits.target))

digits.images.shape

print(digits.images[0])

plt.imshow(digits.images[0],cmap='binary')
plt.show()

print (digits.target.shape)
print (digits.target)

def plot_multi(i):
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()

plot_multi(0)

y = digits.target
x = digits.images.reshape((len(digits.images), -1))
x.shape

x[0]

x_train = x[:1000]
y_train = y[:1000]
x_test = x[1000:]
y_test = y[1000:]
print(x_test.shape)
print(digits.images.shape)
print(digits.images[0].shape)

from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(50, 30, 20), max_iter=1000)

mlp = MLPClassifier(hidden_layer_sizes=(15,), verbose=True)


mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
predictions[:50] 
# we just look at the 1st 50 examples in the test sample

y_test[:50] 
# true labels for the 1st 50 examples in the test sample

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

def plot_multi_pred(i, images, pred, nplots = 16):
    fig = plt.figure(figsize=(10,10))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(images[i+j].reshape(8, 8), cmap='binary')
        plt.title(pred[i+j])
        plt.axis('off')
    plt.show()
plot_multi_pred(0, x_test, predictions)


from PIL import Image as img
import os

i = 0
listdir = os.listdir(os.getcwd() + "/Images")
image_arrays = np.array([[[0 for i in range(8)] for i in range(8)] for i in range(len(listdir))])
for filename in listdir:
    image = img.open(os.path.join(os.getcwd()+ "/Images", filename))
    image_arrays[i] = np.array(image)
    i += 1

image_arrays[1]

image_import = image_arrays.reshape((len(image_arrays), -1))
image_import.shape

iamges_predictions = mlp.predict(image_import)
plot_multi_pred(0, image_import, iamges_predictions, nplots = len(image_import))
