import numpy as np
from scipy.misc import imresize
from sklearn import preprocessing


def preprocess_image(img):
  img = np.array(img, dtype=np.uint8)
  img = imresize(img,(28,28))
  img = np.expand_dims(img,axis=0)
  result = np.expand_dims(img,axis=3)
  return result

def decoder(prediction):
  x = np.argmax(prediction, axis = 1)
  label_encoder = preprocessing.LabelEncoder()
  label_encoder.fit(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                     'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                     '0','1','2','3','4','5','6','7','8','9'])
  result = label_encoder.inverse_transform([x])
  if len(prediction) == 1:
    result = result[0]
  return result
