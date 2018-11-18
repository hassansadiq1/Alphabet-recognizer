# Alphabet-recognizer
Predict english alphabets
Clone this repository
https://github.com/hassansadiq1/Alphabet-recognizer.git

Run following code
import numpy as np
from keras.models import load_model
from scipy.misc import imresize
model = load_model('my_model.h5')

def preprocess_image(img):  
  img = imresize(img,(28,28))
  img = np.expand_dims(img,axis=0)
  result = np.expand_dims(img,axis=3)
  return result

image = None#load image
pre_image = preprocess_image(image)
y = model.predict(pre_image)
print(y)

