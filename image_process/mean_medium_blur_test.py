
# coding: utf-8

# In[3]:

### Import source
from skimage import transform,io
import numpy as np
lena_512 = io.imread('image\\lena512.bmp', as_gray=False)
lena_256 = transform.resize(lena_512, (256, 256))
lena_noise = transform.resize(lena_512, (256, 256)).copy()
for i in range(1024):    
    x=np.random.randint(0,256)
    y=np.random.randint(0,256)
    lena_noise[x,y,:]=1.0
for i in range(1024):
    x=np.random.randint(0,256)
    y=np.random.randint(0,256)
    lena_noise[x,y,:]=0.0


# In[4]:

### Mean and medium blur test
import image_process.convolution as convolution
import numpy as np
import matplotlib.pyplot as plt

_mask = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]])
mean_img = convolution.mean_blur_2d(img=lena_noise,mask=_mask)
medium_img = convolution.medium_blur_2d(img=lena_noise,mask=_mask)

fig = plt.figure(figsize=(13, 13))
plt.subplot(2,2,1),plt.imshow(lena_256,cmap=plt.cm.gray)
plt.title('lena'),plt.axis('off')
plt.subplot(2,2,2),plt.imshow(lena_noise,cmap=plt.cm.gray)
plt.title('salt & pepper noise'),plt.axis('off')
plt.subplot(2,2,3),plt.imshow(mean_img,cmap=plt.cm.gray)
plt.title('mean_img'),plt.axis('off')
plt.subplot(2,2,4),plt.imshow(medium_img,cmap=plt.cm.gray)
plt.title('medium_img'),plt.axis('off')
plt.show()

