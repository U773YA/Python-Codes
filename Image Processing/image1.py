from PIL import Image
import numpy as np
arr=Image.open('C://Users/D-24/Pictures/29fca4f6507b768ce2817813f4d9da0d.jpg')
image=np.array(arr)
image.shape
image2=image+1
data=Image.fromarray(image2,'RGB')
data.show()
image3=255-image
data=Image.fromarray(image3,'RGB')
data.show()
