# Import the imdb data, limit 10000 to keep the top 10000 most occuring words
import matplotlib.pyplot as plt
from keras import models, layers
import numpy as np
from keras.datasets import imdb

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)


print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
print(train_data[0])
print(train_labels[0])


# Preparing the data. Turn lists into tensors
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train[0])

# Vectorize the labels
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asanyarray(test_labels).astype("float32")

# Building  network
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000, )))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))


# Compile the model
model.compile(optimizer="rmsprop", loss="binary_crossentrophy",
              metrics=["accuracy"])


# validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# training model
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy", metrics=["acc"])

history = model.fit(partial_x_train, partial_y_train, epochs=20,
                    batch_size=512, validation_data=(x_val, y_val))


# plot training and validation loss

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc = history_dict["acc"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

# plotting the trainign and validation accuracy
plt.clf()
val_acc = history_dict["val_acc"]

plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# As seen in the chart above, the validation accuracy does not increase after epochs 4,
# Overfits - we are overoptimizing on the training data and end up learning representations
# that are specific to the training data and don't generalize to data outside of the training
# set.

# Retrain the model from scratch, with 4 epochs/iterations
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000, )))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)


print(results)

predictions = model.predict(x_test)

print(predictions)
