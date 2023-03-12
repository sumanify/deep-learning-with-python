
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras import models, layers
import numpy as np
from keras.datasets import reuters

(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))

print(train_data[10])
# Preparing the data


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Sse buil5-in way: from keras.utils.np_utils import to_categorial or following


# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results

# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# Building model


model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

# Compile the model

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy", metrics=["accuracy"])


# Validating the apprach, divide train data into validation and training set
x_val = x_train[:1000]  # take first 1000 to validate
partial_x_train = x_train[1000:]  # use remaining to train

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Train the model
history = model.fit(partial_x_train, partial_y_train, epochs=20,
                    batch_size=512, validation_data=(x_val, y_val))

# Plot the validation and trianing loss


loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# Plotting the training and vlaidation accuracy
plt.clf()

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]


plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# As seen in the above plot, network beginds to overfit after 9 epochs (iteration).
# Retrain a new network from scratch

model = models.Sequential()

model.add(layers.Dense(64, activation="relu", input_shape=(10000, )))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(partial_x_train, partial_y_train, epochs=9,
          batch_size=512, validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)

print(results)


# Generating prediction for the new data

predictions = model.predict(x_test)

print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))
