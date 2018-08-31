from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
arr=Image.open('C://Users/D-24/Pictures/1.jpg')
image=np.array(arr)
image.shape
image2=np.zeros((image.shape[0],image.shape[1],image.shape[2]))
c=50
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        for k in range(0,image.shape[2]):
            image2[i,j,k]=c*math.log(1+image[i,j,k])
            if image2[i,j,k]>255 :
                image2[i,j,k]=255
plt.imshow((image2).astype(np.uint8))
#plt.imshow(image2)
plt.show()
