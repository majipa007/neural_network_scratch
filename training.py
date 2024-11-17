from tensorflow.keras.datasets import mnist
from loss import Loss_CategoricalCrossentropy
from Layer_Dense import Layer_Dense
from model import Model
from activation import *


def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return X_train, y_train, X_test, y_test


def train_network(model, X_train, y_train, epochs=5, batch_size=128, learning_rate=0.1):
    loss_function = Loss_CategoricalCrossentropy()

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')

        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            # Forward pass
            model.forward(batch_X)

            # Calculate loss
            loss = loss_function.calculate(model.layers[-1].output, batch_y)

            # Calculate accuracy
            predictions = np.argmax(model.layers[-1].output, axis=1)
            accuracy = np.mean(predictions == batch_y)

            if i % 1000 == 0:
                print(f'Batch {i}/{len(X_train)}, ' +
                      f'Loss: {loss:.4f}, ' +
                      f'Accuracy: {accuracy:.4f}')

            # Backward pass
            loss_function.backward(model.layers[-1].output, batch_y)
            current_dvalues = loss_function.dinputs

            # Backward pass through all layers
            for layer in reversed(model.layers):
                layer.backward(current_dvalues)
                current_dvalues = layer.dinputs

            # Update weights and biases
            for layer in model.layers:
                if isinstance(layer, Layer_Dense):
                    layer.update(learning_rate)

    return model


def main():
    # Load and preprocess MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Create and configure the model
    model = Model(input_shape=784)
    model.add(Layer_Dense(784, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Train the model
    model = train_network(
        model, X_train, y_train,
        epochs=5,
        batch_size=128,
        learning_rate=0.1
    )

    # Save the model
    model.save('mnist_model')

    # Test accuracy before saving
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f'Test accuracy before saving: {accuracy:.4f}')

    # # Load the model and test again
    # loaded_model = Model.load('mnist_model')
    # predictions = loaded_model.predict(X_test)
    # accuracy = np.mean(predictions == y_test)
    # print(f'Test accuracy after loading: {accuracy:.4f}')


if __name__ == "__main__":
    main()