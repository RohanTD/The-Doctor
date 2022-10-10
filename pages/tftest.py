from keras.models import load_model
import cv2
import numpy as np

model = load_model('lyme.h5')



img = cv2.imread('erythema migrans2.jpg')
img = cv2.resize(img,(128,128))
img = np.reshape(img,[1,128,128,3])

classes = model.predict(img)

print(classes)