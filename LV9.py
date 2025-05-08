from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


train_ds = image_dataset_from_directory(
directory='gtsrb/Train',
labels='inferred',
label_mode='categorical',
batch_size=32,
subset="training",
seed=123,
validation_split=0.2,
image_size=(48, 48))


validation_ds = image_dataset_from_directory(
directory='gtsrb/Train',
labels='inferred',
label_mode='categorical',
batch_size=32,
subset="validation",
seed=123,
validation_split=0.2,
image_size=(48, 48))


test_ds = image_dataset_from_directory(
directory='gtsrb/Test',
labels='inferred',
label_mode='categorical',
batch_size=32,
image_size=(48, 48))


model = models.Sequential([
layers.Conv2D(32, (3,3), activation='relu', padding= 'same',  input_shape=(48,48,3)),
layers.Conv2D(32, (3,3), activation='relu', padding= 'valid',  input_shape=(48,48,3)),
layers.MaxPooling2D((2,2)),
layers.Dropout(0,2),
layers.Flatten(),
layers.Dense(43, activation='softmax')
])

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])


model.fit(train_ds, epochs =4, validation_data=validation_ds, batch_size=32)

pred = model.predict(test_ds)
y_pred_classes = np.argmax(pred, axis=1)
y_true = np.argmax(test_ds, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrica zabune')
plt.xlabel('Predviđeno')
plt.ylabel('Stvarno')
plt.show()