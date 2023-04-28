# Signal that the script started working
with open('train-status.txt', 'w') as file: file.write('0')
print('Python job started...')

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import gc
import sys

# Load the data
data_train = np.load(sys.argv[1], allow_pickle=True) # sys.argv[1]: the file name passed through command line (usually by the Bash script)
data = pd.DataFrame(data_train, columns=['features','labels'])

# Preprocessing
for i in data.index:
    data['features'][i] = data['features'][i].reshape(226, 226, 3)

from keras.utils import to_categorical

# Get the input data
X = np.stack(data['features'].to_numpy())

# Normalize
X = X / 255.0

# Get the labels
y = data['labels'].to_numpy()
y = y.astype(int)

# One-hot encode
y = np.vectorize(lambda x: x//5 if x>1 else x)(y)

# Subtract the minimum label value to ensure that label values start from 0
y = y - np.min(y)

# One-hot encode
y = to_categorical(y, num_classes=3)

# Split the data into training and testing sets (with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print('Performed train-test split!')

# Some memory cleaning
del X
del y
del data
del data_train
gc.collect()

# Load the latest model from the H5 file
model_cnn = load_model('cnn.h5')
model_cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
print('Loaded and compiled the model.')

# Final preparation
X_train, y_train, X_test, y_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), tf.convert_to_tensor(X_test), tf.convert_to_tensor(y_test)

# Train the model
print('About to train!')
model_cnn.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
print('Finished training!')

# Save the model
import os
model_cnn.save(os.path.join(os.getcwd(), 'cnn.h5'))

# Signal that the script finished
print('Python job finished!')
with open('train-status.txt', 'w') as file: file.write('1')