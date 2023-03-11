from keras import models, layers
import numpy as np
from keras.datasets import reuters

(train_data, train_label), (test_data,
                            test_label) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))

# Preparing the data


def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros(len(len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
        return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Sse buil5-in way: from keras.utils.np_utils import to_categorial or following


def to_one_hot(labels, dimensions=46):
    results = np.zeroes(len(labels), dimensions)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_label)
one_hot_test_labels = to_one_hot(test_label)

# Building model


model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(46, activation="softmax"))

# Compile the model

model.compile(optimizer="rmsprop",
              loss="catergorial_crossentrophy", metrics=["accuracy"])
