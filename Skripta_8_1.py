from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

model = models.Sequential([
layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
layers.MaxPooling2D((2,2)),
layers.Conv2D(64, (3,3), activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])

checkpoint = callbacks.ModelCheckpoint('best_model.h5',
monitor='val_accuracy',
save_best_only=True,
verbose=1)
tensorboard_callback = callbacks.TensorBoard(log_dir="logs")

history = model.fit(x_train_s, y_train_s,
epochs=10,
batch_size=64,
validation_split=0.1,
callbacks=[checkpoint, tensorboard_callback])

best_model = models.load_model('best_model.h5')

train_loss, train_acc = best_model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = best_model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")


y_pred = best_model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_s, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrica zabune')
plt.xlabel('Predviđeno')
plt.ylabel('Stvarno')
plt.show()
