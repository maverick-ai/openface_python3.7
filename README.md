# openface_python3.7
Implemented the openface(keras) in pyhton 3.7 and tensorflow 2.0. The weight of the pre trained from orignal openface does not work on python 3.7. 

Load the weights in keras and start predicting the embedding.
face is the image of the face for which we need to find the embedding


from tensorflow.keras.models import load_model

import tensorflow as tf

model = load_model('facenet3.7.h5')

face=cv2.resize(face, (96,96), interpolation = cv2.INTER_AREA)

img = face[...,::-1]

img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)

img = np.transpose(img, (1, 2, 0))

x = np.array([img])

embedding=model.predict(x)
