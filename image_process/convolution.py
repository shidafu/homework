'''
function of image process
author: Leon, date: 2017.12.09
'''
import numpy as np

'''
Operate functions
'''
def operate_mul(x: np.array, y: np.array) -> float:
    return np.sum(np.multiply(x,y))
def operate_max(x: np.array, mask: np.array) -> float:
    return np.max(np.multiply(x,mask))
def operate_min(x: np.array, mask: np.array) -> float:
    return np.min(np.multiply(x,mask))
def operate_medium(x: np.array, mask: np.array) -> float:
    list=[]
    for j in range(len(mask)):
        for i in range(len(mask[0])):
            if mask[j,i]>0:
                list.append(x[j,i])
    return np.median(np.array(list))
def operate_mean(x: np.array, mask: np.array) -> float:
    return np.sum(np.multiply(x,mask))/np.sum(mask)

'''
filters
'''
EDGE_HSOBEL = np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]]) / 4.0
EDGE_VSOBEL = EDGE_HSOBEL.T

EDGE_HSCHARR = np.array([[ 3,  10,  3],
                    [ 0,   0,  0],
                    [-3, -10, -3]]) / 16.0
EDGE_VSCHARR = EDGE_HSCHARR.T

EDGE_HPREWITT = np.array([[ 1, 1, 1],
                     [ 0, 0, 0],
                     [-1,-1,-1]]) / 3.0
EDGE_VPREWITT = EDGE_HPREWITT.T

EDGE_PROBERTS = np.array([[1, 0],
                       [0, -1]], dtype=np.double)
EDGE_NROBERTS = np.array([[0, 1],
                       [-1, 0]], dtype=np.double)

def BLUR_GAUSSIAN(sigma:'Deviation of the gaussian envelope',
                  gamma:'Spatial aspect ratio of the gaussian envelope'=1.0,
                  theta:'Rotation of the gaussian envelope'=0.0,
                  nstds:'int:Number of sigma in bounding box'=2):
    # References:[1] https: // en.wikipedia.org / wiki / Gabor_filter
    #            [2]https: // en.wikipedia.org / wiki / Gaussian_function
    sigma_x = sigma
    sigma_y = float(sigma) * gamma
    # Bounding box
    xmax = np.max([np.abs(nstds * sigma_x * np.cos(theta)), np.abs(nstds * sigma_y * np.sin(theta))])
    xmax = np.ceil(np.max([1, xmax]))
    ymax = np.max([np.abs(nstds * sigma_x * np.sin(theta)), np.abs(nstds * sigma_y * np.cos(theta))])
    ymax = np.ceil(np.max([1, ymax]))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gs = np.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))
    gs = gs / np.sum(gs)
    return gs

def FEA_GABOR(theta:'Rotation of the sin/cos wave',
              lamda:'Wavelength of the sin/cos wave',
              sigma:'Deviation of the gaussian envelope',
              gamma:'Spatial aspect ratio of the gaussian envelope'=1.0,
              psi:'Offset of the sin/cos wave'=0.0,
              nstds:'int:Number of sigma in bounding box'=3):
    # References:[1] https: // en.wikipedia.org / wiki / Gabor_filter
    #            [2]https: // en.wikipedia.org / wiki / Gaussian_function
    sigma_x = sigma
    sigma_y = float(sigma) * gamma
    # Bounding box
    xmax = np.max([np.abs(float(nstds) * sigma_x * np.cos(0)), np.abs(float(nstds) * sigma_y * np.sin(0.5*np.pi))])
    xmax = np.ceil(np.max([1, xmax]))
    ymax = np.max([np.abs(nstds * sigma_x * np.sin(0.5 *np.pi)), np.abs(nstds * sigma_y * np.cos(0))])
    ymax = np.ceil(np.max([1, ymax]))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    real = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))\
         * np.cos(2 * np.pi / lamda * x_theta + psi)
    imag = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2))\
         * np.sin(2 * np.pi / lamda * x_theta + psi)
    return real,imag

'''
2D cov with operation defined by custom
'''
def cov_2d(img : '2D list or np.array',
           filter : '2D list or np.array',
           operation : 'Operate function' = operate_mul,
           padding :('VAlID','SAME') = 'VAlID',
           strides: '[sy,sx]' = [1,1]) -> '2d np.array':
    # para process
    img_array = np.array(img).copy()
    if img_array.ndim == 3:
        if img_array.shape[2] == 1:
            return cov_2d(img = img_array[:,:,0],
                           filter = filter,
                           operation = operation,
                           padding = padding ,
                           strides = strides )
        else:
            rgb=[]
            for cl in range(img_array.shape[2]):
                r = cov_2d(img = img_array[:,:,cl],
                           filter = filter,
                           operation = operation,
                           padding = padding ,
                           strides = strides )
                rgb.append(r)
            return np.dstack(tuple(rgb))
    assert img_array.ndim == 2
    hx = len(img_array)
    wx = len(img_array[0])
    f_array= np.array(filter)
    assert f_array.ndim == 2
    hf = len(f_array)
    wf = len(f_array[0])
    assert hasattr(operation,'__call__')
    assert padding == 'VAlID' or 'SAME'
    assert isinstance(strides,list)
    assert len(strides) == 2
    # do padding
    hx_ = hx
    wx_ = wx
    if padding == 'SAME':
        hy = np.ceil(float(hx+0.0) / float(strides[0]))
        wy = np.ceil(float(wx+0.0) / float(strides[1]))
        hx_ = (hy-1) * strides[0] + hf
        wx_ = (wy-1) * strides[1] + wf
        tf = (hx_ - hx)//2
        lf = (wx_ - wx)//2
        x_array_ex = np.zeros((hx_,wx_))
        x_array_ex[tf:tf+hx,lf:lf+wx] = img_array
    else:
        hy = np.ceil(float(hx-hf+1) / float(strides[0]))
        wy = np.ceil(float(wx-wf+1) / float(strides[1]))
        x_array_ex = img_array
    # do convolution
    y_array= np.zeros((int(hy),int(wy)))
    for j in range(len(y_array)):
        for i in range(len(y_array[0])):
            y_array[j,i] = operation(x_array_ex[j*strides[0]:j*strides[0]+hf,
                                                i*strides[1]:i*strides[1]+wf],
                                     f_array)
    return y_array

'''
2D typical filter operations
'''
def filter_2d(img : '2D list or np.array',
              filter: ('hsobel','vsobel','hscharr','vscharr','hprewitt','vprewitt','proberts','nroberts') = 'hsobel',
              padding : ('VAlID','SAME') = 'VAlID') -> '2d np.array':
    y = None
    if filter == 'hsobel':
        y = cov_2d(img=img, filter=EDGE_HSOBEL, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'vsobel':
        y = cov_2d(img=img, filter=EDGE_VSOBEL, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'sobel':
        h = cov_2d(img=img, filter=EDGE_HSOBEL, operation=operate_mul, padding=padding, strides=[1, 1])
        v = cov_2d(img=img, filter=EDGE_VSOBEL, operation=operate_mul, padding=padding, strides=[1, 1])
        y = np.sqrt(h ** 2 + v ** 2)
        y /= np.sqrt(2)
    elif filter == 'hscharr':
        y = cov_2d(img=img, filter=EDGE_HSCHARR, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'vscharr':
        y = cov_2d(img=img, filter=EDGE_VSCHARR, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'scharr':
        h = cov_2d(img=img, filter=EDGE_HSCHARR, operation=operate_mul, padding=padding, strides=[1, 1])
        v = cov_2d(img=img, filter=EDGE_VSCHARR, operation=operate_mul, padding=padding, strides=[1, 1])
        y = np.sqrt(h ** 2 + v ** 2)
        y /= np.sqrt(2)
    elif filter == 'hprewitt':
        y = cov_2d(img=img, filter=EDGE_HPREWITT, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'vprewitt':
        y = cov_2d(img=img, filter=EDGE_VPREWITT, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'prewitt':
        h = cov_2d(img=img, filter=EDGE_HPREWITT, operation=operate_mul, padding=padding, strides=[1, 1])
        v = cov_2d(img=img, filter=EDGE_VPREWITT, operation=operate_mul, padding=padding, strides=[1, 1])
        y = np.sqrt(h ** 2 + v ** 2)
        y /= np.sqrt(2)
    elif filter == 'proberts':
        y = cov_2d(img=img, filter=EDGE_PROBERTS, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'nroberts':
        y = cov_2d(img=img, filter=EDGE_NROBERTS, operation=operate_mul, padding=padding, strides=[1, 1])
    elif filter == 'roberts':
        h = cov_2d(img=img, filter=EDGE_PROBERTS, operation=operate_mul, padding=padding, strides=[1, 1])
        v = cov_2d(img=img, filter=EDGE_NROBERTS, operation=operate_mul, padding=padding, strides=[1, 1])
        y = np.sqrt(h ** 2 + v ** 2)
        y /= np.sqrt(2)
    else:
        y = None
    return y

'''
2D gaussian blur
'''
def gaussian_blur_2d(img : '2D list or np.array',
                     sigma:'Deviation of the gaussian envelope',
                     gamma:'Spatial aspect ratio of the gaussian envelope'=1.0,
                     theta:'Rotation of the gaussian envelope'=0.0,
                     nstds:'int:Number of sigma in bounding box'=2,
                     padding : ('VAlID','SAME') = 'VAlID') -> '2d np.array':
    y = cov_2d(img=img, filter=BLUR_GAUSSIAN(sigma,gamma,theta,nstds), operation=operate_mul, padding=padding, strides=[1, 1])
    return y

'''
2D medium blur
'''
def medium_blur_2d(img : '2D list or np.array',
                   mask: 'Mask envelope',
                   padding : ('VAlID','SAME') = 'VAlID') -> '2d np.array':
    y = cov_2d(img=img, filter=mask, operation=operate_medium, padding=padding, strides=[1, 1])
    return y

'''
2D mean blur
'''
def mean_blur_2d(img : '2D list or np.array',
                 mask: 'Mask envelope',
                 padding : ('VAlID','SAME') = 'VAlID') -> '2d np.array':
    y = cov_2d(img=img, filter=mask, operation=operate_mean, padding=padding, strides=[1, 1])
    return y

'''
2D gabor transform
'''
def gabor_2d(img : '2D list or np.array',
             theta:'Rotation of the sin/cos wave',
             lamda:'Wavelength of the sin/cos wave',
             sigma:'Deviation of the gaussian envelope',
             gamma:'Spatial aspect ratio of the gaussian envelope'=1.0,
             psi:'Offset of the sin/cos wave'=0.0,
             nstds:'int:Number of sigma in bounding box'=3,
             padding : ('VAlID','SAME') = 'VAlID') -> '2d np.array':
    gabor_r,gabor_i = FEA_GABOR(theta, lamda, sigma, gamma, psi, nstds)
    y_r = cov_2d(img=img, filter=gabor_r, operation=operate_mul, padding=padding, strides=[1, 1])
    y_i = cov_2d(img=img, filter=gabor_i, operation=operate_mul, padding=padding, strides=[1, 1])
    return y_r,y_i