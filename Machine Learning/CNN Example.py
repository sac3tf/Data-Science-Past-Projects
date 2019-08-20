from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

''' CNN MNIST digits classification
3-layer CNN for MNIST digits classification 
First 2 layers - Conv2D-ReLU-MaxPool
3rd layer - Conv2D-ReLU-Dropout
4th layer - Dense(10)
Output Activation - softmax
Optimizer - Adam
~99.4% test accuracy in 10epochs
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
import matplotlib.pyplot as plt

#begin by building intuition with this link for CNNs: https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2

# load mnist dataset
# Split it into training and test data sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))
print("\n\nThe number of the labels is: ", num_labels)

# convert to one-hot vector, that is length 10 to include each y value for each label.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# input image dimensions
image_size = x_train.shape[1]
print("\n\nOriginal Training image shape = ", image_size)


x_train.shape #The original shape is 60000 observations, that are 28 by 28 images
x_test.shape  #The original shape is 10000 observations, that are 28 by 28 images


# resize and normalize
# we are going from dimensions with height and width to height, width, and channels
# we could include more channels if we were using color images, e.g. 3 for Red, Green, Blue (RGB)
# However, since we only have grayscale images, 1 channel is sufficient
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])

#We show here that we have added the additional channel dimension
#This means we technically have a 3D tensor,  a multidimensional tensor
#However, the convolutions will still be 2D since we evaluating just 1 channel.
x_train.shape #The original shape is 60000 observations, that are 28 by 28 images, by 1 channel
x_test.shape  #The original shape is 10000 observations, that are 28 by 28 images, by 1 channel

#Then, we normalize by dividing by 255 to reduce noise and potential for overfitting.
#The intuition for why we scale is to reduce potential for overfitting with a smaller range of values
#Recall from the MLP example that we normalized by the number of grayscale values, (0,255), so 255.

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# network parameters
# image is processed as is (square grayscale)
input_shape = (image_size, image_size, 1)
batch_size = 128 # Recall, batch size should be between 50 to 256.
                 # The batches are used in mini batching to update the error during an 
                 # epoch instead of waiting until the end of the training epoch when we have
                 # evaluated all training images.

#Reference this link for types of convolutions, we'll look over a few: https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
kernel_size = 3  # "In the MLP Model the number of units characterizes the Dense layers,
                 # the kernel characterizes the CNN operations." 
                 # Essentially, a kernel is a rectangular window that slides through the
                 # whole image from left to right and top to bottom. With the size of a kernel,
                 # (in this case 3 by 3), the kernel begins with the top three rows of the image 
                 # - starting at the top left of the image. When the kernel reaches the end of the 
                 # last column of pixel values on the right side of the image, the kernel repositions 
                 # in the left most pixel column and drops down one row. It should be noted that the 
                 # kernel cannot go beyond the border of an image. This sliding of the kernel is known as a convolution.
                 # This continues until we reach the bottom right corner of the image and completes 
                 # one convolution for one image. Convolutions create transform the input image into feature maps.
                 # Feature maps are transformed into new feature maps in successive CNN layers.
                 # It should also be noted that after convolutions, images become smaller.
                 # If you had a 5x5 image with a 3x3 kernel size, the output feature map would be 3x3.
                 # If the dimensions of the input should be the same as the output feature maps, Conv2D
                 # will accapt the option "padding='same'". Then the input is padded with zeroes around the borders
                 # to keep the dimensions unchanged after the convolution.
                 
pool_size = 2    # "The last big change in CNNs is the addition of a "MaxPooling2D" layer with the 
                 # argument "pool_size=2". "MaxPooling2D" compresses each feature map. Every patch
                 # of the size "pool_size"*"pool_size" is reduced by one pixel. The value is equal 
                 # equal to the maximum pixel value within the patch." 
                 # The main reason we use "MaxPooling2D" is the reduction of feature maps size 
                 # which give us increased kernel coverage. So if we have a pool_size=2 and 26x26 input
                 # feature map, the output after MaxPooling2D would be a 13x13 feature map.
                 # We also can use different types of pooling operations. For example, to achieve
                 # the same 50% reduction in input feature map size, instead of "MaxPooling2D(2),
                 # we can use "AveragePooling2D(2)" to get the average of each 2x2 patch in the feature
                 # map. We can also use "Strided colvolution, Conv2D(strides=2, ...)" to get the same 50%
                 # input feature map size reduction for the output feature map size.
                 
                 # Also note that "both pool_size and kernel can be non-square. In these cases, both
                 # the row and column sizes must be indicated. For example, pool_size=(1, 2) and kernel=(3,5).
                 
                 # The output of the last "MaxPooling2D is a stack of feature maps. The role of Flatten is to 
                 # convert the stack of feature maps into a vector format suitable for either Dropout or Dense layers, 
                 # similar to the MLP model output layer."

filters = 64     # The number of feature maps created per Conv2D is controlled by the filters argument.
                 # per the link below, "By default, the filters W are initialised randomly using the 
                 # glorot_uniform method, which draws values from a uniform distribution with positive and negative bounds"
                 # src: https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
                 
dropout = 0.2    # Dropout is a form of regularization, that make neural networks more robust to new unseen test input data
                 # Dropout is not used in the output layer and is only used during model training.
                 # Dropout is not present when making predictions on test data.

# model is a stack of CNN-ReLU-MaxPooling
model = Sequential() # We first call the Keras, sequential API.

#Keras convolutional layers documentation https://keras.io/layers/convolutional/
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu',
                 input_shape=input_shape)) #for the first Conv2D, we specify the input_shape of the images. 
                                           #The other arguments will stay the same in subsequent Conv2D layers.
                                           # The 28 x28 images are reduced into 26x26 feature maps.
model.add(MaxPooling2D(pool_size))         #This MaxPooling2D reduces the 26x26 feature maps to 13x13. There are 64 feature maps.

##########################################################################################################
###### We break momentarily before the next Conv2D layer to show the 64 feature maps at this point #######
##########################################################################################################

#To see the 64 feature maps at this point for any image in the test dataset, change the number below.
img = x_test[4000]

# expand dimensions so that it represents a single 'sample'
img = np.expand_dims(img, axis=0)

# get feature map for first hidden layer
feature_maps = model.predict(img)

# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show()

##########################################################################################################
######                             We return to the second Conv2D layer                            #######
##########################################################################################################

model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))      # The 13x13 feature maps are reduced to 11x11 feature maps        

model.add(MaxPooling2D(pool_size))        # The 11x11 feature maps are reduced to 5x5 feature maps. There are 64 feature maps.

model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 activation='relu'))      # The 5x5 feature maps are reduced into 3x3 feature maps. There are 64 feature maps.

model.add(Flatten())                      # The 64 3x3 feature maps are flattened into vectors 
                                          # of length 576 (3*3*64 = 576).
                                          # Now that the feature maps are flattened, we can use dropout 
                                          # as for regularization to prevent overfitting.
# dropout added as regularizer
model.add(Dropout(dropout))              #(1 - 0.20) * 576 = 461 hidden unit
# output layer is 10-dim one-hot vector
model.add(Dense(num_labels))             # Then we map to the output length for the number of labels, 10.
                                         # this is the output for one-hot vector
model.add(Activation('softmax'))         # softmax squashes the outputs to predicted probabilities of 
                                         # each class that sum to 1. The highest probability 
model.summary()
plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# train the network
model.fit(x_train, y_train, epochs=10, batch_size=batch_size)

#CNNs are more parameter efficient and have higher accuracy than MLPs
#Furthermore, CNNs are more suitible for learning from sequential data (like images and video.)
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))