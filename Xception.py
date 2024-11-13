#!/usr/bin/env python
# coding: utf-8

# # Import TensorFlow and other necessary libraries

# In[1]:


# Importing necessary libraries for data manipulation and analysis
import numpy as np
np.random.seed(0)

# Importing necessary libraries for image manipulation
import PIL

# Importing TensorFlow for machine learning
import tensorflow as tf
tf.random.set_seed(0)

# Importing necessary modules from Keras for building the model
from tensorflow.keras import Input, layers, Sequential



import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


# # Split folders with files(e.g. images) into training, validation and test (dataset) folders.

# In[2]:


import splitfolders # or import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# Training, validation, and testing set
splitfolders.ratio("data", output="plant_images", seed=1337, ratio=(.7, .2, .1), group_prefix=None) # default values


# In[3]:


# import image data for the folder plant_images
import pathlib
data_dir = "plant_images"
data_dir = pathlib.Path(data_dir).with_suffix('')
# Total images in the dataset
image_count = len(list(data_dir.glob('*/*/*.jpg')))
print("Total image in dataset = ",image_count)


# # Here are some plants

# In[4]:


Army_worm = list(data_dir.glob('*/Army_worm/*'))
PIL.Image.open(str(Army_worm[0]))


# In[5]:


Healthy = list(data_dir.glob('*/Healthy/*'))
PIL.Image.open(str(Healthy[0]))


# In[ ]:





# # Load data using a Keras utility

# ## Create a dataset

# In[6]:


# Define some parameters for the loader:
batch_size = 32
img_height = 180
img_width = 180
data_dir = 'data'


# In[7]:


# It's good practice to use a validation split when developing your model. Use 80% of the images for training and 20% for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir ,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[8]:


val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[9]:


class_names = train_ds.class_names
print(class_names)


# # Visualize the data

# In[10]:


# Here are the first 16 images from the training dataset:
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))  # Adjust the size of the figure to accommodate 16 images
for images, labels in train_ds.take(1):
  for i in range(16):  # Change the range to 16
    ax = plt.subplot(4, 4, i + 1)  # Change the subplot parameters to create a 4x4 grid
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[11]:


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# # Configure the dataset for performance

# In[12]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# # Standardize the data

# In[13]:


# Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
normalization_layer = layers.Rescaling(1./255)


# In[14]:


# There are two ways to use this layer. You can apply it to the dataset by calling Dataset.map
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# # Xception model for image classification

# In[15]:


# Create the model
Xception_model = tf.keras.applications.Xception(
    include_top=False,
    input_shape=(180, 180, 3),
    weights='imagenet',
    pooling='avg',
    classes=6
)

# Freeze the weights of the pre-trained model
for layer in Xception_model.layers:
    layer.trainable = False

# Add your custom layers
model = Sequential()
model.add(Xception_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(6, activation='softmax'))


# In[16]:


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# In[17]:


# Model summary
model.build((None, 180, 180, 3))
model.summary()


# In[18]:


from keras.callbacks import EarlyStopping

# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
epochs=200
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs, 
  callbacks=[early_stopping]
)


# # Visualize training results

# In[19]:


# Create plots of the loss and accuracy on the training and validation sets
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

# Plot for Training and Validation Accuracy
plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Plot for Training and Validation Loss
plt.figure(figsize=(8, 8))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[20]:


# Convert to percentage, round to two decimal places, and print the final accuracy values
print('Final training accuracy:', round(acc[-1] * 100, 2), "%")
print('Final validation accuracy:', round(val_acc[-1] * 100, 2), "%")


# In[ ]:




