# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 08:34:44 2021

@author: chomi
"""

from detecto import core, utils
import os
import pandas as pd
import cv2 as cv
import random as r
import matplotlib.pyplot as plt

cdmodel = core.Model.load('crucible.pth', ['sample'])
csv = pd.DataFrame()
path = 'C:/Users/chomi/Desktop/test/'
counter = 0

#auto labelling images in folders
def func(num):
    global counter
    global csv
    for fname in os.listdir(path + str(num) + '/'):
        if(fname.endswith('.png')):
            img = cv.imread(path + str(num) + '/' + fname)
            lbl, box, score = cdmodel.predict(img)
            temp = box.numpy()[0]
            data = {'filename':[fname], 'width':[img.shape[1]], 'height':[img.shape[0]],
                    'class':[str(num)+'_'], 'xmin':[round(temp[0])], 'ymin':[round(temp[1])],
                    'xmax':[round(temp[2])], 'ymax':[round(temp[3])], 'image_id':[counter]}
            csv = csv.append(pd.DataFrame(data))
            # images must be in one directory to be trained. So copying them into common dir
            cv.imwrite(path+fname, img)
            counter+=1
            print('class:', num, 'image_id:', counter)

for i in range(5):
    func(i)

#selecting 10% of images for validation and making new val_label.csv
val = pd.DataFrame()
while(True):
    csv = csv.reset_index(drop=True)
    num = r.randint(0, len(csv)-1)
    row = csv.loc[num]
    csv = csv.drop(num)
    val = val.append(row)
    # moving the validation images to different folder    
    os.rename(path+row['filename'], path+'validation/'+row['filename'])
    if(len(csv) / len(val) <= 9):
        break
val = val[['filename', 'width', 'height','class','xmin','ymin','xmax','ymax','image_id']]

#resetting image_id so they're consecutive
csv = csv.reset_index(drop=True)
for i in range(len(csv)):
    csv.at[i, 'image_id'] = i
val = val.reset_index(drop=True)
for i in range(len(val)):
    val.at[i, 'image_id'] = i

#training
csv.to_csv(path+'labels.csv', index = False, header = True)
val.to_csv(path+'val_label.csv', index = False, header = True)

train_dataset = core.Dataset(path+'labels.csv', path)
val_dataset = core.Dataset(path+'val_label.csv', path+'validation')

loader = core.DataLoader(train_dataset, batch_size=2, shuffle=True)
model = core.Model(['0_','1_','2_','3_','4_'])

losses = model.fit(loader, val_dataset, epochs=10, learning_rate=0.01, verbose=True)
plt.plot(losses)
plt.show()
model.save(path + 'new_model.pth')




























