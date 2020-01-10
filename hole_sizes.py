import os

from imageio import imread
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import measure
from scipy.ndimage import morphology
from sklearn import metrics
import numpy as np



def hole_size(img):

    l = threshold_otsu(img)
    seg = img > l
    full = morphology.binary_fill_holes(seg)
    holes = np.logical_and(full, np.logical_not(seg))

    holes = measure.label(holes)
    a = measure.regionprops(holes)
    area = 0
    for c in a:
        area = area + c.area
    return area


if __name__ == "__main__":

    net = "ZERO-ROT-VGG-UNET"
    projs = 32

    src = "E:\\Nuvem Google\\PythonProjects\\public-fcn-unet\\results-{}-{}-projs\\".format(net, projs)

    samples = len(os.listdir(src))
    values = np.zeros((samples,2))
    i = 0

    for folder in os.listdir(src):

        rec = imread(src + folder + "\\final_rec.png", pilmode="F")
        gt = imread(src + folder + "\\original.png", pilmode="F")

        hs_rec = hole_size(rec)
        hs_gt = hole_size(gt)

        values[i, 0] = hs_rec
        values[i, 1] = hs_gt
        i = i + 1

    values = values[values[:,1].argsort()]
    rmse = metrics.mean_squared_error(values[:,1], values[:,0], squared=False)
    r2 = metrics.r2_score(values[:,1], values[:,0])
    print("RMSE: {}".format(rmse))
    print("R2: {}".format(r2))

    for a in range(samples):
       print("{} {}".format(int(values[a,0]),int(values[a,1])))