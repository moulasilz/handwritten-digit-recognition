import matplotlib.pyplot as plt 
import numpy as np
import idx2numpy
from sklearn.utils import Bunch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image as img
import os
import pickle

digits = Bunch(train_images = 'Datasets/train-images-idx3-ubyte', 
              train_labels = 'Datasets/train-labels-idx1-ubyte',
              test_images = 'Datasets/t10k-images-idx3-ubyte',
              test_labels = 'Datasets/t10k-labels-idx1-ubyte')

train_images = idx2numpy.convert_from_file(digits.train_images)
train_labels = idx2numpy.convert_from_file(digits.train_labels)
test_images = idx2numpy.convert_from_file(digits.test_images)
test_labels = idx2numpy.convert_from_file(digits.test_labels)

print (len(train_images))
print (train_labels[:10])

def plot_multi(i):
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(train_images[i+j], cmap='binary')
        plt.title(train_labels[i+j])
        plt.axis('off')
    plt.show()
plot_multi(0)

x_train = train_images.reshape((len(train_images), -1))
y_train = train_labels
x_test = test_images.reshape((len(test_images), -1))
y_test = test_labels
print(x_train.shape)
print(x_test.shape)

mlp = MLPClassifier(hidden_layer_sizes=(30,20,10), max_iter=1000, verbose=True)

mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
predictions[:50] 
y_test[:50] 


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
plot_multi_pred(x_test, predictions)

accuracy_score(y_test, predictions)

# dự đoán hình ảnh mẫu 
i = 0
listdir = os.listdir(os.getcwd() + "/Images/28x28")

image_arrays = np.array([[[0 for i in range(28)] for i in range(28)] for i in range(len(listdir))])
for filename in listdir:
    image = img.open(os.path.join(os.getcwd()+ "/Images/28x28", filename))
    image_arrays[i] = np.array(image)
    i += 1

image_import = image_arrays.reshape((len(image_arrays), -1))
image_import.shape

iamges_predictions = mlp.predict(image_import)
plot_multi_pred(images= image_import, pred=iamges_predictions, nplots = 16)

# # lưu và load model đã qua huấn luyện
# pickle.dump(mlp, open('model.pkl','wb'))
# model = pickle.load(open('model.pkl','rb'))

# predict = model.predict(x_test[100:126])
# plot_multi_pred(images=x_test[100:126], pred=predict)

# dự đoán hình ảnh Paint thực tế
listdir = os.listdir(os.getcwd() + "/static/uploads")
i = 0
upload_image_arrays = np.array([[[0 for i in range(28)] for i in range(28)] for i in range(len(listdir))])
for filename in listdir:
    image = img.open(os.path.join(os.getcwd()+ "/static/uploads", filename))
    if image.size != [28,28]:
        image = image.resize((28,28))
    gray = np.array(image)[:,:,0]
    upload_image_arrays[i] = np.array(gray)
    i += 1
    
upload_image_arrays = upload_image_arrays.reshape((len(upload_image_arrays), -1))
upload_image_arrays.shape

upload_image_preds = mlp.predict(upload_image_arrays)
plot_multi_pred(images=upload_image_arrays, pred=upload_image_preds, nplots=len(upload_image_preds))

