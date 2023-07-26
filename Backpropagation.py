import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from scipy.special import expit

# Define the Backpropagation Neural Network class
class BackpropagationNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly with mean 0
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize biases as zeros
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
        
    def forward_propagation(self, X):
        # Compute the output of the neural network
        self.z1 = np.dot(X, self.W1) + self.b1 
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward_propagation(self, X, y, output):
        # Compute the gradients of weights and biases
        self.delta2 = (output - y) * self.sigmoid_derivative(self.z2)
        self.dW2 = np.dot(self.a1.T, self.delta2)
        self.db2 = np.sum(self.delta2, axis=0, keepdims=True)
        
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.sigmoid_derivative(self.z1)
        self.dW1 = np.dot(X.T, self.delta1)
        self.db1 = np.sum(self.delta1, axis=0)
        
    def update_parameters(self, learning_rate):
        # Update weights and biases using the gradients
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        
    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Backward propagation
            self.backward_propagation(X, y, output)
            
            # Update parameters
            self.update_parameters(learning_rate)
            
            # Print the loss every 100 epochs
            # if epoch % 10 == 0:
            loss = np.mean(np.square(y - output))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def sigmoid(self, x):
        return expit(x)
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def load_model(self, model_file):
        # Load the model from a .h5 file
        self.model = load_model(model_file)
        self.W1, self.b1, self.W2, self.b2 = self.model.get_weights() 

# Load and preprocess your tomato leaf image dataset
model_file = 'C:/Users/user/Dataset80%100epoch.h5'

# Define the root directory where your folders are located
root_directory = 'D:\\Kuliah\\Skripsi\\Dataset\\tomato\\test'

# List of valid image file extensions
valid_extensions = ['.jpg', '.jpeg', '.png']

# Initialize empty lists for images and their corresponding labels
images = []
labels = []

# Traverse through the root directory and its subdirectories
for root, dirs, files in os.walk(root_directory):
    # Iterate over the files in the current directory
    for file in files:
        # Check if the file has a valid image extension
        _, ext = os.path.splitext(file)
        if ext.lower() in valid_extensions:
            # Construct the file path
            file_path = os.path.join(root, file)
            
            # Read the image using PIL (or any other image processing library)
            img = load_img(file_path, target_size=(224, 224))
            img_tensor = img_to_array(img)
            
            # Append the image and its corresponding label to the lists
            images.append(img_tensor)
            labels.append(root.split('\\')[5])  # Use the directory name as the label or customize as needed

# Convert images and labels to numpy arrays
X = np.array(images)
y = np.array(labels)

# Flatten the images and normalize the pixel values
X = X.reshape(X.shape[0], -1) / 255.0

# Encode labels using one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# Define the network dimensions
input_size = X.shape[1]  # Update according to your image size
hidden_size = 3  # Adjust according to your preference
output_size = len(label_encoder.classes_)  # Number of disease categories

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=42)

# Create an instance of the BackpropagationNN class
nn = BackpropagationNN(input_size, hidden_size, output_size)

# Train the network
learning_rate = 0.01
epochs = 100
nn.train(X_train, y_train, learning_rate, epochs)

# Test the network
predictions = nn.forward_propagation(X_test)