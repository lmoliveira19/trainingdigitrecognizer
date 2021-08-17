# 1. Install Dependencies
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("keras")
install("tensorflow")

# 2. Load MNIST Datasets of Digital hand writing
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Print Predicted digits vector
print (y_train)


print ("X_train - training input size {}".format(X_train.shape))
print ("y_train - training result size {}\n".format(y_train.shape))

print ("X_test - Validation input size {}".format(X_test.shape))
print ("y_test - Validation result size {}".format(y_test.shape))

# E.g. Plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# Show the plot
plt.show()


# 3. Create a simple Artificial Neural Network for Training
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# 3.1 Load Data

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_trainOrig = X_train
X_testOrig = X_test


# 3.2 Preprocess Data - Flatten 

# Flatten 28x28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2] # 28 * 28 = 784 pixels
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

print ("Training matrix size X_train = {}, test matrix size X_test = {}".format(X_train.shape, X_test.shape))


# 3.3 Preprocess Data - Normalize Pixel values

# Normalize inputs from 0-255 to 0-1
print ("Max pixel value before Normalization = {}".format(X_train.max())) # 255 is the max

X_train = X_train / 255
X_test = X_test / 255

print ("Max pixel value post Normalization= {}".format(X_train.max())) # 1 is the max


# 3.4 Preprocess Data - Convert Ground truth to One hot encoding

# One hot encode outputs (i.e. convert number into array, i.e. 5 - [0 0 0 0 1 0 0 0 0 0] to handle one vs all)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1] # Now shape is 10000 x 10 as each number is converted as array

print ("Number of classes - {}".format(num_classes))

# 3.6 build the model

# Model Creation Class
# simple neural network with one hidden layer with the same number of neurons as there are inputs (784). A rectifier activation function is used for the neurons in the hidden layer. 
# A softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. 
# Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.

# Define the base model
def baseline_model(num_pixels, num_classes):
    # Create Model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation ='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation = 'softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model            

model = baseline_model(num_pixels, num_classes)


# 3.7 Train the Model

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 1, batch_size=200, verbose=2)


# 3.8 Evaluate the Model against Test Data
# Test data and validation is used as the same. But, this should be ideally different

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print (scores)
print ("Baseline Error: %.2f%%" % (100-scores[1]*100))


# 3.9 Save the Model
import os
modelFilePath = "kerasDigitRecognizer.h5"
if os.getenv('MODEL_PATH'):
    modelFilePath = str(os.getenv('MODEL_PATH')) + "/kerasDigitRecognizer.h5"
model.save(modelFilePath)


# 4 Sample visual Test: Picking one of the inputs from Training data

testInput = X_trainOrig[506]
plt.imshow(testInput, cmap=plt.get_cmap('gray'))
plt.show()
# Preprocess - Flatten
nrPixels = testInput.shape[0] * X_train.shape[1] # 28 * 28 = 784 pixels
testInput_processed = testInput.reshape(1, num_pixels).astype('float32')
# Preprocess - Normalization
testInput_processed = testInput_processed/ 255
# Prediction
pred = model.predict(testInput_processed)
print ("Prediction for the input image is {}".format(pred.argmax()))
