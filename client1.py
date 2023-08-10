import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def getRandomSamplesOfData(quantity_of_labels,x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<quantity_of_labels[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1

    sns.displot(dy, discrete=True)
    plt.xlim((-1,10))
    plt.show() 
    return np.array(dx), np.array(dy)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
quantity_of_labels = sys.argv[2].split(',')
quantity_of_labels = np.asarray(quantity_of_labels, dtype=int)
x_train, y_train = getRandomSamplesOfData(quantity_of_labels,x_train, y_train)


model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)