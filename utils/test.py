import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from tensorflow import keras

model = load_model('model.h5')

model = load_model('model.h5')
image_path = './insti/test/positive/44.jpg'
image = keras.utils.load_img(
    image_path, grayscale=False, color_mode='rgb', target_size=None,
    interpolation='nearest'
)
img = np.array(image)
img = img / 255.0
img = img.reshape(1, 224, 224, 3)
label = model.predict(img)
print("Predicted Class (0 - Negative, 1- Positive): ")

if label >= 0.5:
    print("Positive")
else:
    print("Negative")
