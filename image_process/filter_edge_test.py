
# coding: utf-8

# In[1]:

### Import source
from skimage import transform,io
lena_512 = io.imread('image\\lena512.bmp', as_gray=True)
lena_256 = transform.resize(lena_512, (256, 256))


# In[2]:

### Edge filters test
import image_process.convolution as convolution
import matplotlib.pyplot as plt
filters=['hsobel','vsobel','sobel', 'hscharr','vscharr','scharr',
         'hprewitt','vprewitt','prewitt', 'proberts','nroberts','roberts']
fig = plt.figure(figsize=(12, 20))
plt.subplot(5,3,1),plt.imshow(lena_256,cmap=plt.cm.gray)
plt.title('lena'), plt.axis('off')
for i in range(len(filters)):
    y = convolution.filter_2d(img=lena_256,filter=filters[i])
    plt.subplot(5,3,i+4),plt.imshow(y,cmap=plt.cm.gray)
    plt.title(filters[i]), plt.axis('off')
plt.show()