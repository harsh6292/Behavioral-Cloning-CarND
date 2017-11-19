#import keras
import csv
import cv2
import numpy as np

DBG = True

lines = []

# Load Udacity training data
udacity_training_log_file = 'udacity_data/driving_log.csv'

if (DBG):
	print(udacity_training_log_file)

with open(udacity_training_log_file) as csvfile:
	reader = csv.reader(csvfile)
	
	# Read each line in driving_log.csv
	count = 0
	for line_in_file in reader:
		# Udacity driving_log.csv file has first line as column name not actual data which gives error while training
		if count == 0:
			count = 1
			continue

		lines.append(line_in_file)

# Append my own training data to udacity's training data
own_training_log_file = 'own_training_data/driving_log.csv'

with open(own_training_log_file) as train_csvfile:
	reader = csv.reader(train_csvfile)

	# Read each line in driving_log.csv
	for line_in_file in reader:
		lines.append(line_in_file)


images = []
measurements = []

i =0

img_data_dir = 'udacity_data/'


# Method to get image file and store it using opencv
def process_image(img_path):
	filename = img_path.split('/')[-1]
	current_path = img_data_dir + 'IMG/' + filename
	image = cv2.imread(current_path)

	return image


# Process all the images in driving_log (left, center, right)
# Add the steering measurements and store it

for line in lines:
	if DBG and i < 2:
		print('Processing line: {}'.format(line))
	
	# Extract hood image path from each line (Center image)
	img_center = process_image(line[0])
	img_left = process_image(line[1])
	img_right = process_image(line[2])

	if DBG and i < 1:
		print("Image shape from opencv: {}".format(img_center.shape))

	steer_angle_center = float(line[3])
	i += 1

	# Ignore most of the straight angles
	if steer_angle_center < 0.1 and steer_angle_center > -0.1:
		continue

	# Save the opencv image to a list for processing later	
	images.append(img_center)
	images.append(img_left)
	images.append(img_right)

	# Extract steering angle from each line
	#steer_angle_center = float(line[3])

	if DBG and i < 2:
		print("Measurement from file: {}".format(steer_angle_center))

	# Steering correction angle
	correction = 0.067
	steer_angle_left = steer_angle_center + correction
	steer_angle_right = steer_angle_center - correction

	# Add measurement to a list of measurements
	measurements.append(steer_angle_center)
	measurements.append(steer_angle_left)
	measurements.append(steer_angle_right)

if DBG:
	print("Total images: {}, total measurements: {}".format(len(images), len(measurements)))

# Image augmentation using flipped images
augmented_images, augmented_measurements = [], []


# Apply image augmentation by flipping the images and measurements
# This helps to reduce overfitting as most of the data is left-turning biased
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)

	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement*-1.0)


if DBG:
	print("Total augmented images: {}, total augmented measurements: {}".format(len(augmented_images), len(augmented_measurements)))
	
# Convert all images read through opencv to numpy arrays, our training data
X_train = np.array(augmented_images)

# Create label array as numpy array usin steering angle measurements
y_train = np.array(augmented_measurements)

print('Total training data: Input shape: {}, Label shape: {}'.format(X_train.shape, y_train.shape))

# Build model using keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Cropping2D

# Build a Sequential model with convolution and dense layers
model = Sequential()

# Add input layer
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

#Add Lambda normalization layer
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Layer-1 Convolution
# Filter size = 16, kernel_size = 5x5, padding = valid, 
model.add(Convolution2D(8, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)	, strides=None, border_mode='valid'))

# Layer-2 convolution
model.add(Convolution2D(12, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))

# Layer-3 convolution
model.add(Convolution2D(16, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))


# Use either globalpooling layer or flatten layer
#model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(1500))

model.add(Dense(300))

model.add(Dense(1))


# Print the summary of the model
model.summary()

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the data to model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)


# Save the model
model.save('model.h5')


# End
