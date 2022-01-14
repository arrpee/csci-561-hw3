#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.metrics import ConfusionMatrixDisplay


#%%
# Data and variable initializations
rng = np.random.default_rng()

x_train = np.loadtxt("data/train_image.csv", dtype=np.float64, delimiter=",")
y_train = np.loadtxt("data/train_label.csv", dtype=int, delimiter=",")
x_test = np.loadtxt("data/test_image.csv", dtype=np.float64, delimiter=",")
y_test = np.loadtxt("data/test_label.csv", dtype=int, delimiter=",")
#%%
# Data preprocessing
x_train /= 255.0
x_test /= 255.0

# %%
class NeuralNetwork:
    def __init__(self, shape, num_epochs, learning_rate, mu, batch_size):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mu = mu
        self.batch_size = batch_size
        self.shape = shape

        self.weights = [
            rng.normal(size=(shape[i], shape[i + 1]))
            * np.sqrt(1.0 / (shape[i] + shape[i + 1]))
            for i in range(len(shape) - 1)
        ]
        self.activations = [
            np.zeros((self.batch_size, shape[i + 1],)) for i in range(len(shape) - 1)
        ]
        self.biases = [np.zeros(shape[i + 1]) for i in range(len(shape) - 1)]
        self.velocities = [
            np.zeros((shape[i], shape[i + 1])) for i in range(len(shape) - 1)
        ]

    def _batch(self, data, labels):
        for i in range(0, data.shape[0], self.batch_size):
            yield data[i : i + self.batch_size, :], labels[i : i + self.batch_size]

    def _relu(self, activations):
        return activations * (activations > 0)

    def _sigmoid(self, activations):
        return 1 / (1 + np.exp(-activations))

    def _softmax(self, activations):
        temp = np.exp(activations)
        return temp / np.sum(temp, axis=1, keepdims=True)

    def _cross_entropy(self, labels):
        return -np.sum(
            np.log(self.activations[-1][np.arange(self.batch_size), labels])
            / self.batch_size,
            axis=0,
        )

    def _forward_pass(self, batch):
        self.activations[0] = self._relu(
            np.dot(batch, self.weights[0]) + self.biases[0]
        )

        for i in range(1, len(self.shape) - 2):
            self.activations[i] = self._relu(
                np.dot(self.activations[i - 1], self.weights[i]) + self.biases[i]
            )

        self.activations[-1] = self._softmax(
            np.dot(self.activations[-2], self.weights[-1]) + self.biases[-1]
        )

    def _compute_loss(self, batch_labels):
        self.loss = self._cross_entropy(batch_labels)

    def _backward_pass(self, batch_labels):
        gradient = self.activations[-1]
        gradient[np.arange(self.batch_size), batch_labels] -= 1
        gradient = gradient / self.batch_size

        hidden = np.dot(gradient, self.weights[-1].T)
        hidden[self.activations[-2] < 0] = 0

        temp = self.velocities[-1].copy()
        temp2 = self.velocities[-2].copy()

        self.velocities[-1] = self.mu * self.velocities[
            -1
        ] - self.learning_rate * np.dot(self.activations[-2].T, gradient)
        self.velocities[-2] = self.mu * self.velocities[
            -2
        ] - self.learning_rate * np.dot(self.activations[-3].T, hidden)

        self.weights[-1] += -self.mu * temp + (1 + self.mu) * self.velocities[-1]
        self.weights[-2] += -self.mu * temp2 + (1 + self.mu) * self.velocities[-2]

        self.biases[-1] -= self.learning_rate * np.sum(gradient, axis=0)
        self.biases[-2] -= self.learning_rate * np.sum(hidden, axis=0)

        for i in range(3, len(self.shape) - 1):
            hidden = np.dot(hidden, self.weights[-i + 1].T)
            hidden[self.activations[-i] < 0] = 0
            temp = self.velocities[-i].copy()
            self.velocities[-i] = self.mu * self.velocities[
                -i
            ] - self.learning_rate * np.dot(self.activations[-i - 1].T, hidden)

            self.weights[-i] += -self.mu * temp + (1 + self.mu) * self.velocities[-i]
            self.biases[-i] -= self.learning_rate * np.sum(hidden, axis=0)

    def fit(self, data, labels):
        training_accuracy = []
        validation_accuracy = []
        test_accuracy = []

        temp = rng.random(data.shape[0])
        indices = temp < np.percentile(temp, 90)
        self.training_set = data[indices]
        self.validation_set = data[~indices]
        self.training_labels = labels[indices]
        self.validation_labels = labels[~indices]

        for i in tqdm(range(self.num_epochs)):
            for batch, batch_labels in self._batch(
                self.training_set, self.training_labels
            ):
                self._forward_pass(batch)
                self._compute_loss(batch_labels)
                self._backward_pass(batch_labels)

            if i % 5 == 1:
                self.learning_rate /= 2

            self.predict(self.training_set)
            training_accuracy.append(
                sum(np.argmax(self.activations[-1], axis=1) == self.training_labels)
                / self.training_set.shape[0]
            )
            self.predict(self.validation_set)
            validation_accuracy.append(
                sum(np.argmax(self.activations[-1], axis=1) == self.validation_labels)
                / self.validation_set.shape[0]
            )

            self.predict(x_test)
            test_accuracy.append(
                sum(np.argmax(self.activations[-1], axis=1) == y_test) / x_test.shape[0]
            )

            clear_output(wait=True)
            plt.plot(
                list(range(len(training_accuracy))),
                training_accuracy,
                label="Training",
            )
            plt.plot(
                list(range(len(validation_accuracy))),
                validation_accuracy,
                label="Validation",
            )
            plt.plot(
                list(range(len(test_accuracy))), test_accuracy, label="Test",
            )
            plt.xlabel(
                f"Train:{training_accuracy[-1]:.2f} Val:{validation_accuracy[-1]:.2f} Test:{test_accuracy[-1]:.2f}"
            )
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()

    def predict(self, data):
        self._forward_pass(data)
        return np.argmax(self.activations[-1], axis=1)


network = NeuralNetwork(
    (784, 512, 512, 10),
    num_epochs=50,
    learning_rate=0.01,
    mu=0.95,
    batch_size=min(25, x_train.shape[0], x_test.shape[0]),
)
network.fit(x_train, y_train)

predictions = network.predict(network.training_set)
ConfusionMatrixDisplay.from_predictions(
    network.training_labels, predictions, cmap=plt.cm.Blues,
)


predictions = network.predict(network.validation_set)
ConfusionMatrixDisplay.from_predictions(
    network.validation_labels, predictions, cmap=plt.cm.Blues,
)


predictions = network.predict(x_test)
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, cmap=plt.cm.Blues,
)
