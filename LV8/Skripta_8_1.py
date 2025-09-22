from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# TODO: definiraj callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
# TODO: provedi treniranje mreze pomocu .fit()
history = model.fit(
    x_train_s, y_train_s,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=[tensorboard_cb, checkpoint_cb]
)

#TODO: Ucitaj najbolji model
best_model = keras.models.load_model("best_model.h5")

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = best_model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = best_model.evaluate(x_test_s, y_test_s, verbose=0)

print(f"Točnost na trening skupu: {train_acc:.4f}")
print(f"Točnost na test skupu: {test_acc:.4f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
_train_pred = np.argmax(best_model.predict(x_train_s), axis=-1)
cm_train = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predikcija")
plt.ylabel("Stvarna vrijednost")
plt.title("Matrica zabune - Trening skup")
plt.show()

# Matrica zabune – na test skupu
y_test_pred = np.argmax(best_model.predict(x_test_s), axis=-1)
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predikcija")
plt.ylabel("Stvarna vrijednost")
plt.title("Matrica zabune - Test skup")
plt.show()