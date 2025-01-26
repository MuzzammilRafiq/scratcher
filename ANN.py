import numpy as np
import pandas as pd


class Linear:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.B = np.zeros(1)  # Initialize bias to zero
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.B

    def backward(self, dE_dY, lr):
        dE_dW = self.X.T @ dE_dY
        dE_dB = np.sum(dE_dY)
        dE_dX = dE_dY @ self.W.T

        # Update parameters
        self.W -= lr * dE_dW
        self.B -= lr * dE_dB

        return dE_dX


class TanH:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward(self, dE_dY, _lr):
        sigma_bar = 1 - np.tanh(self.X) ** 2
        return dE_dY * sigma_bar  # Element-wise multiplication


class MSE:
    def __init__(self):
        self.X = None
        self.O = None

    def forward(self, X, O):
        self.X = X
        self.O = O
        return np.mean((X - O) ** 2)

    def backward(self):
        n = self.X.shape[0]
        return (self.X - self.O) * (2 / n)


# Training Loop Setup
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def train(X_train, y_train, model, loss_fn, lr, epochs, batch_size):
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_loss = 0
        for i in range(0, n_samples, batch_size):
            # Get batch
            X_batch = X_shuffled[i : i + batch_size]
            O_batch = y_shuffled[i : i + batch_size]

            # Forward pass
            activations = X_batch
            for layer in model:
                activations = layer.forward(activations)

            # Compute loss
            loss = loss_fn.forward(activations, O_batch)
            total_loss += loss

            # Backward pass
            grad = loss_fn.backward()
            for layer in reversed(model):
                grad = layer.backward(grad, lr)

        # Print epoch statistics
        avg_loss = total_loss / (n_samples // batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def preprocess(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = np.array(df_train.iloc[:, 1:]) / 255
    y_train = np.array(df_train.iloc[:, 0])
    X_test = np.array(df_test.iloc[:, 1:]) / 255
    y_test = np.array(df_test.iloc[:, 0])
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess(
        "dataset/mnist_train.csv", "dataset/mnist_test.csv"
    )

    lr = 0.01
    epochs = 4
    batch_size = 32
    n_samples = X_train.shape[0]

    model = [
        Linear(784, 256),
        TanH(),
        Linear(256, 256),
        TanH(),
        Linear(256, 256),
        TanH(),
        Linear(256, 10),
    ]
    loss_fn = MSE()
    train(X_train, y_train, model, loss_fn, lr, epochs, batch_size)
