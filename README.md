---

# Neural Network from Scratch

This repository contains a fully functional implementation of a neural network framework built from scratch using Python and NumPy. It includes essential components such as dense layers, activation functions, loss functions, and a model architecture that supports saving and loading. The project is designed to provide a foundational understanding of how neural networks work under the hood, without relying on high-level libraries like TensorFlow or PyTorch for the core operations.

---

## 🚀 **Features**
- Implements the forward and backward propagation algorithm.
- Modular design for layers, activation functions, and loss functions.
- Support for training on the MNIST dataset for digit classification.
- Custom training loop with options for batch size, learning rate, and epochs.
- Model serialization and deserialization (save/load models).
- Configurable architecture, allowing easy addition of layers and activations.

---

## 📁 **Project Structure**
The repository is organized as follows:

```plaintext
├── training.py              # Main script for training and evaluating the model
├── model.py                 # Model class for managing architecture and weights
├── Layer_Dense.py           # Dense (fully connected) layer implementation
├── activation.py            # Activation functions (ReLU, Softmax)
├── loss.py                  # Loss functions (Categorical Crossentropy)
├── README.md                # Project documentation
```

---

## 📊 **How It Works**

1. **Model Construction**:
   - Define a `Model` instance and add layers such as `Layer_Dense`, `Activation_ReLU`, and `Activation_Softmax`.

2. **Training**:
   - Train the model on the MNIST dataset using a custom training loop.
   - Monitor the loss and accuracy during training for insights.

3. **Saving and Loading**:
   - Save the trained model, including architecture and weights, to a specified directory.
   - Load saved models for inference or further training.

---

## 🔧 **Setup**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/majipa007/neural_network_scratch.git
   cd neural_network_scratch
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed, along with NumPy and TensorFlow for the MNIST dataset.

   ```bash
   pip install numpy tensorflow
   ```

3. **Run the Training Script**:
   ```bash
   python training.py
   ```

---

## 🧠 **How to Use**

1. **Load Data**:
   The MNIST dataset is automatically downloaded and preprocessed in the `training.py` script.

2. **Customize Model**:
   Modify the architecture in the `main()` function of `training.py`:
   ```python
   model = Model(input_shape=784)
   model.add(Layer_Dense(784, 128))
   model.add(Activation_ReLU())
   model.add(Layer_Dense(128, 10))
   model.add(Activation_Softmax())
   ```

3. **Train the Model**:
   Configure training parameters like `epochs`, `batch_size`, and `learning_rate` in the `train_network` function.

4. **Save/Load Model**:
   Save your model using:
   ```python
   model.save('path_to_save_directory')
   ```
   Load the model with:
   ```python
   model = Model.load('path_to_save_directory')
   ```

---

## 📈 **Results**

- **Test Accuracy**:
  The network achieves a reasonable accuracy on the MNIST test set with the default configuration.

- Example training log:
  ```
  Epoch: 1
  Batch 0/60000, Loss: 2.3026, Accuracy: 0.0781
  ...
  Test accuracy before saving: 0.9200
  ```

---

## 💡 **Key Components**

1. **Layers**:
   - `Layer_Dense`: Fully connected layer with weight and bias updates.
   
2. **Activations**:
   - `ReLU`: Rectified Linear Unit for non-linearity.
   - `Softmax`: Probability distribution for classification.

3. **Loss**:
   - `Categorical Crossentropy`: Suitable for multi-class classification tasks.

4. **Model Management**:
   - Save and load models, including their weights and architecture.

---

## 📖 **Learnings and Insights**
This project demonstrates:
- How to implement forward and backward propagation.
- The importance of gradient-based optimizers and learning rates.
- The mechanics of training a simple neural network from scratch.

---

