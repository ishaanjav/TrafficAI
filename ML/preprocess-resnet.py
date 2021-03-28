import keras
import pandas as pd

resnet=keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224,224,3),
    pooling="avg",
)

import os
import numpy as np
import matplotlib.pyplot as plt
all_frames = os.listdir("frames")
for vid in all_frames:
    input = []
    for frame in vid:
        path = 'frames/'+vid+"/"+frame
        frame = plt.imread(path)
        input.append(frame)
    out=resnet(np.array(input))
    np.save(path+".npy", out)

print(resnet(np.ones([1, 224, 224, 3])))