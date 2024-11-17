from model import Model
from training import *

X_train, y_train, X_test, y_test = load_mnist_data()
# Load the model and test again
loaded_model = Model.load('mnist_model')
predictions = loaded_model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Test accuracy after loading: {accuracy:.4f}')