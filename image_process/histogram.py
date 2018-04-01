'''
function of image process
author: Leon, date: 2017.12.09
'''
import numpy as np

'''
Histogram functions
'''
def get_histo(img: '0.0~1.0 valued 2D/3D list or np.array, if 3d depth == 3',
              level:int = 256) -> 'histo,cdf':
    # para process
    img_array = np.array(img).copy()
    if img_array.ndim == 3:
        if img_array.shape[2] == 1:
            return get_histo(img = img_array[:,:,0],level=level)
        else:
            rgb=[]
            for cl in range(img_array.shape[2]):
                r = get_histo(img = img_array[:,:,cl],level=level)
                rgb.append(r)
            return np.vstack(tuple(rgb))
    assert img_array.ndim == 2
    assert np.max(img_array) <= 1.0
    histo = np.zeros(level,dtype=np.float64)
    cdf = np.zeros(level,dtype=np.float64)
    for j in range(len(img_array)):
        for i in range(len(img_array[0])):
            histo[int(img_array[j,i]*(level-1))] += 1
    cdf[0] = histo[0]
    for i in range(level-1):
        cdf[i+1] = cdf[i] + histo[i+1]
    return histo,cdf

'''
Image mapping
'''
def apply_mapping(img: '0.0~1.0 valued 2D/3D list or np.array, if 3d depth == 3',
                  map: '1D list or np.array, if img is 3d map is tuple',
                  level: int = 256) -> np.array:
    # para process
    img_array = np.array(img).copy()
    if type(map) is tuple:
        assert img_array.ndim == 3
        assert img_array.shape[2] == len(map)
        rgb = []
        for cl in range(len(map)):
            r = apply_mapping(img=img_array[:, :, cl], map=map[cl])
            rgb.append(r)
        return np.dstack(tuple(rgb))
    else:
        assert len(map) == level
        assert np.max(img_array) <= 1.0
        maped = np.zeros_like(img_array)
        for j in range(len(img_array)):
            for i in range(len(img_array[0])):
                maped[j,i] = map[int(img[j,i]*(level-1))]
    return maped


'''
Histogram equalization
'''
def equal_histo(img: '0.0~1.0 valued 2D/3D list or np.array, if 3d depth == 3',
                level:int = 256) -> np.array:
    # References:[1] https: // en.wikipedia.org / wiki / Gabor_filter
    # para process
    img_array = np.array(img).copy()
    if img_array.ndim == 3:
        return 0
    assert img_array.ndim == 2
    assert np.max(img_array) <= 1.0
    histo,cdf = get_histo(img = img,level = level)
    equaled = apply_mapping(img = img, map = cdf/cdf[level-1],level = level)
    return equaled