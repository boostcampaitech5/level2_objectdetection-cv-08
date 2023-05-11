#%% 
from data_loader.dataset import *

#annotation:str, data_dir:str, transforms=None,indices=None

annotation = '/opt/ml/dataset/train.json'
data_dir = '/opt/ml/dataset'

class test():
    def __init__(self, aa):
        self.a = aa
    
def test_fn():
    return 1

import matplotlib.pyplot as plt
import numpy as np

im = np.random.random(256*256).reshape(256,256)
plt.imshow(im)
# %%
