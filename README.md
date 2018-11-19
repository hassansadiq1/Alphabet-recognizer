# Alphabet-recognizer

Clone this repository
https://github.com/hassansadiq1/Alphabet-recognizer.git

Load pretrained model
```python
from keras.models import load_model
model = load_model('alphabets.h5')
```
Import some useful fuctions and perform operations as follows
```python
from alpha_utils.py import preprocess_image,decoder
image = None#load image
pre_image = preprocess_image(image)
y = model.predict(pre_image)
y = decoder(y)
```
Result will be a character or string depending on number of images

