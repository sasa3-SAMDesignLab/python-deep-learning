import sys, os
os.chdir('/Users/sasa./miniforge3/envs/py38/code-list/deep-learning/2_Neural-Network')
sys.path.append('/Users/sasa./miniforge3/envs/py38/code-list/deep-learning')
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)