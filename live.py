import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
from models import *

g = load_model('celebgen64.h5')


noise = np.random.normal(0, 0.65, (1, 25))
noise2 = noise

print(np.amax(noise))
print(np.amin(noise))
print(np.mean(noise))

im = g.predict(noise)
plot = plt.imshow(im[0])

plt.ion()
noise = np.random.normal(0, 0.8, (1, 25))

ind = 0 

while True:
    noise[0][ind] = np.random.normal(0, 0.8,1)
    im = g.predict(noise)
    plot.set_data(im[0])
    ind += 1
    if ind > 24:
        ind = 0
    
    plt.pause(0.2)

plt.ioff()
plt.show()

