import os
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np

from skimage.filters import gaussian

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_nrmse as mse


net = 'ZERO-ROT-UNET-SOTA'
metric = "psnr"

src01 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-1-projs\\'.format(net)
src04 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-4-projs\\'.format(net)
#src08 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-8-projs\\'.format(net)
src16 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-16-projs\\'.format(net)
src32 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-32-projs\\'.format(net)
#src60 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-61-projs\\'.format(net)
#src120 = 'D:\\Nuvem Google\\PythonProjects\public-fcn-unet\\results-{}-121-projs\\'.format(net)

#, src07, src15, src30, src30, src60, src120

projs = [1,4,16, 32]
i = 0

for src in [src01, src04, src16, src32]:
    values = []
    for folder in os.listdir(src):
        example = os.path.join(src,folder)

        files = os.listdir(example)

        sol = misc.imread(os.path.join(example,files[0]), mode="F")
        gt = misc.imread(os.path.join(example,files[2]), mode = "F")

        gt = gt - gt.mean()
        sol = sol - sol.mean()

        #gt = (gt-gt.min())/(gt.max()-gt.min())
        #sol = (sol - sol.min()) / (sol.max() - sol.min())

        if metric == "ssim":
            v = ssim(sol, gt, data_range=gt.max()-gt.min())
        elif metric == "psnr":
            v = psnr(gt, sol, data_range=gt.max()-gt.min())
        else:
            v = mse(gt,sol)
        values.append(v)
    np.save("{}-{}-{}-projs".format(metric,net,projs[i]), values)
    print("{} {} {}".format(projs[i], np.mean(values), np.std(values)))
    i = i + 1
