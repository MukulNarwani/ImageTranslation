import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import random
from tensorflow_examples.models.pix2pix import pix2pix
 

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 10

train_ds,test_ds= tf.keras.utils.image_dataset_from_directory(
  'dataset',
  validation_split=0.2,
  subset='both',
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

def showExamples(ds):
  num_batch_trainds = ds.cardinality().numpy()
  random_batch = random.randint(1,num_batch_trainds)
  
  for i in range(len(ds.class_names)):
    filtered_ds = ds.filter(lambda x, l: tf.math.equal(l[0], i))
    for image, label in filtered_ds.take(5):
          ax = plt.subplot(1,2, i+1)
          plt.imshow(image[0].numpy().astype('uint8'))
          plt.axis('off')
  plt.show()

##--Image Processing--##

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0,:,:,:]
  # print(image[0,:,:,:].shape)
  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = normalize(random_jitter(image))
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

faces = train_ds.filter(lambda x, l: tf.math.equal(l[0], 0))
maps = train_ds.filter(lambda x, l: tf.math.equal(l[0], 1))
faces=faces.map(preprocess_image_train,num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)
maps=maps.map(preprocess_image_train,num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)
test_ds=test_ds.map(preprocess_image_test,num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

##--MODEL--##
OUTPUT_CHANNELS = 3

generator_m = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

##--Training--##

to_face=generator_f.predict(maps)
to_map=generator_m.predict(faces)

generator_f.summary()
