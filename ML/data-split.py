import os
from tqdm import tqdm
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


videos = os.listdir("videos")
annots = json.load(open("annotations.json", "r"))

if not os.path.isdir("train"):
    os.mkdir("train")
if not os.path.isdir("train/crash"):
    os.mkdir("train/crash")
if not os.path.isdir("train/nocrash"):
    os.mkdir("train/nocrash")

for file in tqdm(videos):
    path = os.path.join("videos", file)
    try:
        annot=annots[file]
    except:
        continue
    if (len(annot)==0): continue

    keyframes = [annot[i]['keyframes'] for i in range(len(annot))]
    frame_bounds = []
    tqdm.write(file)
    for frame in keyframes:
        frame_bounds.append((int(frame[0]['frame']), int(frame[1]['frame'])))
    
    try:
        cap = cv2.VideoCapture(path)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = np.empty((frameCount, 224, 224, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            frames[fc] = cv2.resize(frame, (224, 224))
            fc += 1
        cap.release()
    except:
        print(f"ERROR: {file}")
        continue
    crashframes=[]
    for i in frame_bounds:
        crashframes.extend(frames[i[0]:i[1]])
    nocrashframes = np.array(frames[0:frame_bounds[0][0]])
    if nocrashframes.shape[0]==0:
        nocrashframes=np.array([])
    crashframes = np.array(crashframes)
    tqdm.write(str(crashframes.shape))
    tqdm.write(str(nocrashframes.shape))
    for i in range(0, len(crashframes), 3):
        frame = crashframes[i]
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(frame, aspect='auto')
        # plt.show()
        if not os.path.isdir("train/crash/"+file.split(".")[0]):
            os.mkdir("train/crash/"+file.split(".")[0])
        plt.savefig("train/crash/"+file.split(".")[0]+"/"+str(i)+".png")
    for i in range(0, len(nocrashframes), 3):
        frame = nocrashframes[i]
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(frame, aspect='auto')
        # plt.show()
        if not os.path.isdir("train/nocrash/"+file.split(".")[0]):
            os.mkdir("train/nocrash/"+file.split(".")[0])
        plt.savefig("train/nocrash/"+file.split(".")[0]+"/"+str(i)+".png")