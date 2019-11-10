# USAGE
# --dataset
# /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/animals
# --model
# /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/myoutput/simple_nn.h5
# --label-bin
# /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/myoutput/simple_nn_lb.pickle
# --plot
# /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/myoutput/simple_nn_plot.png


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from tensorflow import keras

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

layers = tf.keras.layers
models = tf.keras.models

"""
--dataset /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/animals
--model /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/myoutput/simple_nn.h5
--label-bin /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/myoutput/simple_nn_lb.pickle
--plot /Users/patrickryan/Development/python/mygithub/pyimagesearch-keras-tutorial/myoutput/simple_nn_plot.png

"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

height = 32
width = 32
color_channels = 3

dim = (width, height)
# loop over the input images
for imagePath in imagePaths:
	# load the image, resize the image to be 32x32 pixels (ignoring
	# aspect ratio), flatten the image into 32x32x3=3072 pixel image
	# into a list, and store the image in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, dim).flatten()
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras
model = models.Sequential()

# pyimagesearch original model
# 3072 = 32*32*3
# , kernel_regularizer=keras.regularizers.l1(l=0.1)
model.add(layers.Dense(1024, input_shape=((height * width * color_channels),), activation="sigmoid"))
# model.add(layers.Dropout(0.2))  # add dropout to control overfitting
model.add(layers.Dense(512, activation="sigmoid"))

# my model
# model.add(layers.Dense(1024, input_shape=(3072,), activation="relu"))
# model.add(layers.Dense(512, activation="relu"))
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(512, activation="relu"))
# model.add(layers.Dropout(0.25))
# model.add(layers.Dense(128, activation="relu"))


model.add(layers.Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = tf.keras.optimizers.SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
testY_values = testY.argmax(axis=1)
prediction_values = predictions.argmax(axis=1)
print(f'Accuracy: {accuracy_score(testY_values, prediction_values)}')
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])

f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()