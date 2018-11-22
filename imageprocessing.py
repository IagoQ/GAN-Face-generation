from PIL import Image
import numpy as np
import glob
from math import floor,ceil
from skimage.transform import resize


imageCount = 5000
firstImage = 0
width = 128
height = 128

name = 'celeb'+str(width)+'-'+str(imageCount)
labelName = 'labels'+str(imageCount)
labelText = open('attributes.txt','r')




def getAttributes(line):
    splitted = line.split(' ')
    values = []
    for entry in splitted[1:]:
        if entry is not '':
            values.append(int(entry))
    return np.array(values).reshape((40,1))



lines = labelText.readlines()
lines.pop(0)
lines.pop(0)



images = np.empty((imageCount,height,width,3))
labels = np.empty((imageCount,40,1))


imageNames = glob.glob('images\*')
for i,imageN in enumerate(imageNames[firstImage:]):
    if i == imageCount:
        break
    
    im = np.array(Image.open(imageN))
    print(im.shape)
    im = resize(im,(height,width),preserve_range=True)
    images[i] = im
    labels[i] = getAttributes(lines[i+firstImage])


print(images.shape)
print(labels.shape)
np.save(name,images.astype('uint8'))
np.save(labelName,labels.astype('uint8'))




'''
target = 128

for i in range(len(images)):
    if images[i].shape[0] > images[i].shape[1]:
        aspect = ceil(target/images[i].shape[0] * images[i].shape[1])
        images[i] = resize(images[i],(target,aspect),preserve_range=True)

        padding_r = ceil((target - images[i].shape[1])/2)
        padding_l = floor((target - images[i].shape[1])/2)
        images[i] = np.pad(images[i], ((0, 0), (padding_r, padding_l),(0,0)), 'constant')

    else:
        aspect = ceil(target/images[i].shape[1] * images[i].shape[0])
        images[i] = resize(images[i],(aspect,target),preserve_range=True)

        padding_t = ceil((target - images[i].shape[0])/2)
        padding_b = floor((target - images[i].shape[0])/2)
        images[i] = np.pad(images[i], ((padding_t, padding_b),(0, 0),(0,0)), 'constant')

data = np.array(images).astype('uint8')


np.save('Simpsons_128.npy',data)
print(np.amin(data))
print(np.amax(data))

'''