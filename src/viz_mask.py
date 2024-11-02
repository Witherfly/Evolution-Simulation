import cv2 
import os
import numpy as np 

from utils import get_newest_file




path_wall = get_newest_file("custom_masks/wall_masks/")

path_death = get_newest_file("custom_masks/death_masks/")

if True:
    death_mask = np.loadtxt(path_death) *0.5
    wall_mask = np.loadtxt(path_wall)

    final_mask = death_mask + wall_mask

else:
    
    final_mask = np.loadtxt(path_death)


cv2.imshow("mask", cv2.resize(final_mask, (500, 500)))
cv2.waitKey(0)

