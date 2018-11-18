# It takes grayscale images and predict alphabet
import numpy as np
from keras.models import load_model
from scipy.misc import imresize
model = load_model('.h5')

def preprocess_image(img):  
  img = imresize(img,(28,28))
  img = np.expand_dims(img,axis=0)
  result = np.expand_dims(img,axis=3)
  return result

image = None#load your image
pre_image = preprocess_image(image)
y = model.predict(pre_image)
y = np.argmax(y,axis = 1)
print(y)
