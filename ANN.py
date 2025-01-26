import numpy as np
import pandas as pd


class Linear:
    def __init__(self, input_size, output_size):
        # Initialize weights with small random values and biases with zeros
        self.W = np.random.randn(input_size, output_size) * 0.01  # Weight matrix
        self.B = np.zeros(output_size)  # Bias vector
        self.X = None  # Store inputs for backward pass

    def forward(self, X):
        # Forward pass: Compute WX + B and store input X
        self.X = X
        return np.dot(X, self.W) + self.B

    def backward(self, dE_dY, lr):
        # Backward pass: Compute gradients and update parameters
        dE_dW = self.X.T @ dE_dY  # Gradient for weights
        dE_dB = np.sum(dE_dY, axis=0)  # Gradient for biases (sum over batch)
        dE_dX = dE_dY @ self.W.T  # Gradient for inputs (to pass to previous layer)

        # Update parameters using learning rate
        self.W -= lr * dE_dW
        self.B -= lr * dE_dB

        return dE_dX  # Return gradient for previous layer


class TanH:
    def __init__(self):
        self.X = None  # Store inputs for backward pass

    def forward(self, X):
        # Forward pass: Apply tanh activation and store input
        self.X = X
        return np.tanh(X)

    def backward(self, dE_dY, _lr):
        # Backward pass: Compute gradient of tanh (1 - tanh^2)
        sigma_bar = 1 - np.tanh(self.X) ** 2  # Derivative of tanh
        return dE_dY * sigma_bar  # Element-wise multiplication


class MSE:
    def __init__(self):
        self.X = None  # Store predictions
        self.O = None  # Store targets

    def forward(self, X, O):
        # Compute Mean Squared Error loss
        self.X = X
        self.O = O
        return np.mean((X - O) ** 2)

    def backward(self):
        # Compute gradient of MSE loss
        n = self.X.size  # Total number of elements in batch
        return (self.X - self.O) * (2 / n)


def one_hot(y, num_classes):
    # Convert class labels to one-hot vectors
    return np.eye(num_classes)[y]


def train(X_train, y_train, model, loss_fn, lr, epochs, batch_size):
    n_samples = X_train.shape[0]
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_loss = 0
        num_batches = 0

        # Process mini-batches
        for i in range(0, n_samples, batch_size):
            # Get batch data
            X_batch = X_shuffled[i : i + batch_size]
            O_batch = y_shuffled[i : i + batch_size]

            # Forward pass through all layers
            activations = X_batch
            for layer in model:
                activations = layer.forward(activations)

            # Compute loss
            loss = loss_fn.forward(activations, O_batch)
            total_loss += loss
            num_batches += 1

            # Backward pass through all layers
            grad = loss_fn.backward()
            for layer in reversed(model):
                grad = layer.backward(grad, lr)

        # Print epoch statistics
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def preprocess(train_path, test_path):
    # Load and preprocess MNIST data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Normalize pixel values and one-hot encode labels
    X_train = np.array(df_train.iloc[:, 1:]) / 255.0
    y_train = np.array(df_train.iloc[:, 0])
    X_test = np.array(df_test.iloc[:, 1:]) / 255.0
    y_test = np.array(df_test.iloc[:, 0])

    # Convert labels to one-hot encoding
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = preprocess(
        "dataset/mnist_train.csv", "dataset/mnist_test.csv"
    )

    # Hyperparameters
    lr = 0.01  # Learning rate
    epochs = 4  # Training epochs
    batch_size = 32  # Mini-batch size

    # Define network architecture
    model = [
        Linear(
            784, 256
        ),  # Input layer (784 features) to first hidden layer (256 units)
        TanH(),  # Activation function
        Linear(256, 256),  # Second hidden layer
        TanH(),
        Linear(256, 256),  # Third hidden layer
        TanH(),
        Linear(256, 10),  # Output layer (10 classes)
    ]

    # Initialize loss function
    loss_fn = MSE()

    # Train the model
    train(X_train, y_train, model, loss_fn, lr, epochs, batch_size)
