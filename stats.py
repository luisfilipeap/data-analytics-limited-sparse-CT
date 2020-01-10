import numpy as np
from scipy import stats



x = np.load("mse-ZERO-ROT-UNET-SOTA-15-projs.npy")
y = np.load("mse-ZERO-ROT-VGG-UNET-15-projs.npy")


t, pvalue = stats.wilcoxon(x,y)

print(pvalue)