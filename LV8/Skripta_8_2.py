import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models
import numpy as np

img_original = mpimg.imread('test.png')
img = color.rgb2gray(img_original)
img = resize(img, (28, 28))
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')  
plt.show()

img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

# TODO: ucitaj izgradenu mrezu
model = models.load_model("best_model.h5")


# TODO: napravi predikciju za ucitanu sliku pomocu mreze
red = model.predict(img)
pred_class = np.argmax(pred)


# TODO: ispis rezultat u terminal
print("Predikcija za sliku:", pred_class)
print("Vjerojatnosti po klasama:", pred)

