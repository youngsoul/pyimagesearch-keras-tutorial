# import the necessary packages
from keras import backend as K
import tensorflow as tf


class SmallVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = tf.keras.models.Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # CONV => RELU => POOL layer set
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                                         input_shape=inputShape))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # (CONV => RELU) * 2 => POOL layer set
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # (CONV => RELU) * 3 => POOL layer set
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same"))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization(axis=chanDim))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))

        # softmax classifier
        model.add(tf.keras.layers.Dense(classes))
        model.add(tf.keras.layers.Activation("softmax"))

        # return the constructed network architecture
        return model
