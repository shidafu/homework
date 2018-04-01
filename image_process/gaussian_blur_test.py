
# coding: utf-8

# In[1]:

### Import source
from skimage import transform,io,data
lena_512 = io.imread('image\\lena512.bmp', as_grey=False)
lena_256 = transform.resize(lena_512, (256, 256))


# In[2]:

### Gaussian blur test
import image_process.convolution as convolution
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec 

_sigma = 2.0
_gamma = 0.25
_theta = [0, 0.5*np.pi]
_nstds = 2
blur1 = convolution.BLUR_GAUSSIAN(sigma=_sigma, gamma=_gamma, theta=_theta[0], nstds=_nstds)
blur2 = convolution.BLUR_GAUSSIAN(sigma=_sigma, gamma=_gamma, theta=_theta[1], nstds=_nstds)
gauss1 = convolution.gaussian_blur_2d(img=lena_256,sigma=_sigma, gamma=_gamma, theta=_theta[0], nstds=_nstds)
gauss2 = convolution.gaussian_blur_2d(img=lena_256,sigma=_sigma, gamma=_gamma, theta=_theta[1], nstds=_nstds)

fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(2, 3,height_ratios=[2,5])
plt.subplot(gs[1]),plt.imshow(blur1,cmap=plt.cm.gray)
plt.title('blur'),plt.axis('off')
plt.xlim(-3,5),plt.ylim(0,8)

plt.subplot(gs[2]),plt.imshow(blur2,cmap=plt.cm.gray)
plt.title('blur'),plt.axis('off')
plt.xlim(0,8),plt.ylim(-3,5)

plt.subplot(gs[3]),plt.imshow(lena_256,cmap=plt.cm.gray)
plt.title('lena'),plt.axis('off')

plt.subplot(gs[4]),plt.imshow(gauss1,cmap=plt.cm.gray)
plt.title('blur1'),plt.axis('off')

plt.subplot(gs[5]),plt.imshow(gauss2,cmap=plt.cm.gray)
plt.title('blur2'),plt.axis('off')
plt.show()

