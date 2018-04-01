
# coding: utf-8

# In[1]:

### Import source
from skimage import transform,io,data
lena_512 = io.imread('image\\lena512.bmp', as_grey=True)
lena_256 = transform.resize(lena_512, (256, 256))


# In[2]:

### Edge filters test
import histogram
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

hist1,_ = histogram.get_histo(img=lena_256,level=256)
eqhist_img = histogram.equal_histo(img=lena_256,level=256)
hist2,_ = histogram.get_histo(img=eqhist_img,level=256)


fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2,2,width_ratios=[4,8])
plt.subplot(gs[0]),plt.imshow(lena_256,cmap=plt.cm.gray)
plt.title('lena'), plt.axis('off')

plt.subplot(gs[1]),plt.bar(np.arange(256),hist1)
plt.title('histogram')

plt.subplot(gs[2]),plt.imshow(eqhist_img,cmap=plt.cm.gray)
plt.title('equalized image'), plt.axis('off')

plt.subplot(gs[3]),plt.bar(np.arange(256),hist2)
plt.title('equalized histogram')
plt.show()

