# Import the imdb data, limit 10000 to keep the top 10000 most occuring words
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
print(train_data[0])
print(train_labels[0])


# Preparing the data. Turn lists into tensors
import numpy as np

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

# Building your network
