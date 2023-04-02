from PIL import Image as img
import cv2
import numpy

image = img.open("1680279913.8126523.png")
print(image.size)
print(image)
print(numpy.array(image))

if image.size != [28,28]:
    image = image.resize((28,28))
    
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_array = numpy.array(image)
for i in range(4):
    gray=img_array[:,:,i]
    print(gray)