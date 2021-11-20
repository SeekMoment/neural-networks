import base64
import io

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

model = keras.applications.VGG16()
model.summary()
img = Image.open('cats.jpg')
img = img.resize((224, 224))
img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

res = model.predict( x )
print(np.argmax(res))