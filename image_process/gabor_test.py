
# coding: utf-8

# In[4]:

### Import source
from skimage import transform,io,data
lena_512 = io.imread('image\\lena512.bmp', as_grey=True)
lena_256 = transform.resize(lena_512, (256, 256))


# In[5]:

### Gabor transform test
import image_process.convolution as convolution
import numpy as np
import matplotlib.pyplot as plt

_theta = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi]
_lamda = [4.0,6.0,8.0]
_sigma = [2.0,3.0,4.0]
_gamma = 1.0

fig = plt.figure(figsize=(4, 3))
for l in range(len(_lamda)):
    for t in range(len(_theta)):
        gb_real,gb_imag = convolution.FEA_GABOR(theta=_theta[t],lamda=_lamda[l],sigma=_sigma[l])
        plt.subplot(len(_lamda),len(_theta),l*len(_theta)+t+1)
        plt.imshow(gb_imag,cmap=plt.cm.gray)
        plt.axis('off')
plt.show()


# In[6]:

### Gaussian blur test
fig = plt.figure(figsize=(12, 9))
for l in range(len(_lamda)):
    for t in range(len(_theta)):
        gb_real,gb_imag = convolution.gabor_2d(img=lena_256,theta=_theta[t],lamda=_lamda[l],sigma=_sigma[l])
        plt.subplot(len(_lamda),len(_theta),l*len(_theta)+t+1)
        plt.imshow(gb_imag,cmap=plt.cm.gray)
        plt.axis('off')
plt.show()

