import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

num_classes = 10
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu', input_shape=x_train_s.shape[1:]),
    layers.Conv2D(64, (3,3), strides=1, padding='valid', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2), strides=2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_loss"
)
tensorboard_cb = keras.callbacks.TensorBoard(log_dir="./logs")

history = model.fit(
    x_train_s, y_train_s,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint_cb, tensorboard_cb]
)

test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na test skupu: {test_acc:.4f}")

y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
