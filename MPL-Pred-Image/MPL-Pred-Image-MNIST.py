#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt 
import numpy as np


# In[5]:


# from sklearn import datasets
# digits = datasets.load_digits()

import idx2numpy
import numpy as np
from sklearn.utils import Bunch
digits = Bunch(train_images = 'Datasets/train-images-idx3-ubyte', 
              train_labels = 'Datasets/train-labels-idx1-ubyte',
              test_images = 'Datasets/t10k-images-idx3-ubyte',
              test_labels = 'Datasets/t10k-labels-idx1-ubyte')

train_images = idx2numpy.convert_from_file(digits.train_images)
train_labels = idx2numpy.convert_from_file(digits.train_labels)
test_images = idx2numpy.convert_from_file(digits.test_images)
test_labels = idx2numpy.convert_from_file(digits.test_labels)



# In[6]:


print(test_images[0])


# In[7]:


import random
i = random.randrange(0, len(test_images))
plt.imshow(test_images[i],cmap='binary')
plt.show()
print(test_labels[i])


# In[8]:


print (len(train_images))
print (train_labels)


# In[9]:


def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(train_images[i+j], cmap='binary')
        plt.title(train_labels[i+j])
        plt.axis('off')
    plt.show()
    
plot_multi(0)


# In[10]:


y = len(train_labels)
x = train_images.reshape((len(train_images), -1))
x.shape


# In[11]:


x[0]


# In[12]:


# x_train = x[:1000]
# y_train = y[:1000]
# x_test = x[1000:]
# y_test = y[1000:]
x_train = x
y_train = train_labels
x_test = test_images.reshape((len(test_images), -1))
y_test = test_labels
print(x_test.shape)
print(y_train.shape)

print(x_train.shape)
print(x_train[0].shape)


# In[13]:


from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(50, 30, 20), max_iter=1000)
mlp = MLPClassifier(hidden_layer_sizes=(30,20,10), max_iter=1000, verbose=True)


# In[14]:


mlp.fit(x_train,y_train)


# In[15]:


predictions = mlp.predict(x_test)
predictions[:50] 
# we just look at the 1st 50 examples in the test sample


# In[16]:


y_test[:50] 
# true labels for the 1st 50 examples in the test sample


# In[17]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[29]:


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


# In[19]:


from PIL import Image as img
import os
i = 0
listdir = os.listdir(os.getcwd() + "/Images/28x28")

# for image, label in zip(idx2numpy.convert_from_file(digits.test_images), range(50)):
#     img.fromarray(image).save(os.getcwd()+ "/Images/28x28/" + str(label) + ".jpg")
     
image_arrays = np.array([[[0 for i in range(28)] for i in range(28)] for i in range(len(listdir))])
for filename in listdir:
    image = img.open(os.path.join(os.getcwd()+ "/Images/28x28", filename))
    image_arrays[i] = np.array(image)
    i += 1


# In[20]:


image_arrays[1]


# In[21]:


image_import = image_arrays.reshape((len(image_arrays), -1))
image_import.shape


# In[22]:


iamges_predictions = mlp.predict(image_import)
plot_multi_pred(0, image_import, iamges_predictions, nplots = 16)


# In[32]:


import pickle
pickle.dump(mlp, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
predict = model.predict(x_test[100:126])

plot_multi_pred(images=x_test[100:126], pred=predict)

