from PIL import Image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

img_path = "./test_sign.jpeg"
img = Image.open(img_path)
img = img.convert("L")
img = img.resize((28, 28))
img_array = np.array(img).astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)  
img_array = np.expand_dims(img_array, axis=-1)

model = keras.models.load_model("best_model.h5")

pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]
print(f"Predviđena klasa: {pred_class}")

plt.imshow(img_array[0, :, :, 0], cmap='gray')
plt.title(f"Predviđena klasa: {pred_class}")
plt.axis('off')
plt.show()
