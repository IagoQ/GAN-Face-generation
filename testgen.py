import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from models import *

noise = np.random.normal(0, 1, (10, 25))

print(noise)


g = load_model('testG.h5')

#g = generator()

im = g.predict(noise)


plt.figure(1)
plt.imshow(im[0])

plt.figure(2)
plt.imshow(im[1])

plt.figure(3)
plt.imshow(im[2])
plt.show()
