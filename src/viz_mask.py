import cv2 
import os
import numpy as np 



def newest(dir_path):
    # os.chdir(dir_path)
    files = os.listdir(dir_path)

    key = lambda f : os.path.getctime(os.path.join(dir_path, f))
    max_file = max(files, key=key)
    
    # os.chdir('..')
    return os.path.join(dir_path, max_file)

path_wall = newest("custom_masks/wall_masks/")

path_death = newest("custom_masks/death_masks/")

if True:
    death_mask = np.loadtxt(path_death) *0.5
    wall_mask = np.loadtxt(path_wall)

    final_mask = death_mask + wall_mask

else:
    
    final_mask = np.loadtxt(path_death)


cv2.imshow("mask", cv2.resize(final_mask, (500, 500)))
cv2.waitKey(0)

