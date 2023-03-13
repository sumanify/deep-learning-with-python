import numpy as np
from keras import models, layers
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_tragets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_targets)


# Preparing the data by normalzing it
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# Build the network
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu",
              input_shape=(train_data.shape[1], )))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# validation


k = 4

num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("processing fold #", i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]