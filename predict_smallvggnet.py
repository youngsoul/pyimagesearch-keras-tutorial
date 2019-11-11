# import the necessary packages
import tensorflow as tf
import pickle
import cv2

height = 64
width = 64
# nn_model = 'myoutput/simple_nn_adrian.h5'
# nn_labels = 'myoutput/simple_nn_adrian_lb.pickle'

nn_model = 'myoutput/smallvggnet.h5'
nn_labels = 'myoutput/smallvggnet_lb.pickle'

images = ['images/cat.jpg', 'images/cats_00843.jpg', 'images/dog.jpg', 'images/dogs_00163.jpg', 'images/panda.jpg', 'images/panda_00755.jpg']
lb = pickle.loads(open(nn_labels, "rb").read())
print(f"Labels: {lb.classes_}")

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = tf.keras.models.load_model(nn_model)

for image_path in images:
	# load the input image and resize it to the target spatial dimensions
	image = cv2.imread(image_path)
	output = image.copy()
	image = cv2.resize(image, (width, height))

	# scale the pixel values to [0, 1]
	image = image.astype("float") / 255.0

	# flatten for NN but NOT CNN
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


	# make a prediction on the image
	preds = model.predict(image)
	print(f"Predictions: {preds} for {image_path}")

	# find the class label index with the largest corresponding
	# probability
	i = preds.argmax(axis=1)[0]
	label = lb.classes_[i]

	# draw the class label + probability on the output image
	text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
	cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(0, 0, 255), 2)

	# show the output image
	cv2.imshow("Image", output)
	cv2.waitKey(0)