# %% 
# Imports
import tensorflow as tf
import numpy as np

import time

print('TF Version:' + tf.__version__)

COLLECTION = 'phsi' # defines the name of the folder with the images
TEST_IMAGE = '1808.jpg' # defines the model's input image
PATH = 'root/' + COLLECTION + '/' # defines the set root directory
PATH_MODEL = 'models/' + COLLECTION # defines the path for the trained model

# Function
# Loads image into two tensors (each part of the image, real and input, goes to one tensor)
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  image = tf.cast(image, tf.float32)

  return image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

# Normalizing the images to [-1, 1]
def normalize(image):
  return (image / 127.5) - 1

test_image = load(PATH + TEST_IMAGE)
test_image = normalize(test_image)
test_width = tf.shape(test_image)[1]
test_height = tf.shape(test_image)[0]

OUTPUT_CHANNELS = 3

def generate_and_save_images(model, test_input, index):
  prediction = model(test_input, training=True)
  
  img = prediction[0] * 0.5 + 0.5
  img = tf.keras.preprocessing.image.array_to_img(img)
  img.save(PATH + 'result/result (' + str(index) + ').png')
  
def generate(model, test_input):
  prediction = model(test_input, training=True)
  return prediction[0] * 0.5 + 0.5

# Run the trained model on a few examples from the test dataset
i = 0
totalTestTime = 0;

generator = tf.keras.models.load_model(PATH_MODEL)

result = np.zeros((test_height, test_width, 3))
step = 256
margin = int(step/2)

for i in range(0, test_width, step - margin):
  for j in range(0, test_height, step - margin):
    if (i + 256 <= test_width and j + 256 <= test_height):
      testTime = time.time()
      
      inp = test_image[None, j:(j+256), i:(i+256), :]
      
      generated = generate(generator, inp).numpy()
      
      # result[j:j+256, i:i+256, :] = generated * 0.5
      result[j:j+256, i:i+256, :] += generated * 0.25
      result[j+margin:j+256-margin, i:i+256, :] += generated[margin:256-margin, :, :] * 0.25
      result[j:j+256, i+margin:i+256-margin, :] += generated[:, margin:256-margin, :] * 0.25      
      result[j+margin:j+256-margin, i+margin:i+256-margin, :] = generated[margin:256-margin, margin:256-margin, :]
      
      # if (i > margin and i <= test_width - margin and j > margin and j <= test_height - margin):
      #   result[j+margin:j+256-margin, i+margin:i+256-margin, :] = generated[margin:256-margin, margin:256-margin, :]
      # else:
      #   result[j:j+256, i:i+256, :] = generated
     
      totalTestTime += time.time() - testTime    
    
img = tf.keras.preprocessing.image.array_to_img(result)
img.save(PATH + 'result/' + TEST_IMAGE)

print('Average test time: {}s'.format(totalTestTime))