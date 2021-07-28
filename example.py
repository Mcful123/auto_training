# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:16:07 2021

@author: chomi
"""

from detecto import core, utils, visualize
import os

model = core.Model.load('new_model.pth', ['0_','1_', '2_','3_','4_'])
directory = 'C:/Users/chomi/Desktop/test/'

for filename in os.listdir(directory):
    if(filename.endswith(".png")): # or '.jpg'
        img = utils.read_image(directory + filename)
        pred = model.predict(img)
        lbl, box, score = pred
        visualize.show_labeled_image(img, box, lbl)
        print("score: ", score)
        
     

