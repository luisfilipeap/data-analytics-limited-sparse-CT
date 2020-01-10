import numpy as np
from matplotlib import pyplot as plt


#gnet = np.load('validation_accuracy_GOOGLENET-model-4-projs.npy')
unet = np.load('validation_accuracy_UNET-SOTA-model-16-projs.npy')
our = np.load('validation_accuracy_VGG-UNET-model-16-projs.npy')

#gnet = [x for x in gnet if x > 0]
unet = [x for x in unet if x > 0]
our = [x for x in our if x > 0]



plt.figure()
#plt.plot(gnet)
plt.plot(unet)
plt.plot(our)
plt.show()


for k in range(len(our)):
    print("{} {}".format(k, our[k][0]))

