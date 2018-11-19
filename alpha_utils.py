import numpy as np
from scipy.misc import imresize
from sklearn import preprocessing

def preprocess_image(img):  
  img = imresize(img,(28,28))
  img = np.expand_dims(img,axis=0)
  result = np.expand_dims(img,axis=3)
  return result

def decoder(prediction):
  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(['A','B','C','D','E','F','G','H','I','J','K','L','M',
        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
  result = label_encoder.inverse_transform([prediction])
  if len(prediction) = 1:
    result = result[0]
  return result
