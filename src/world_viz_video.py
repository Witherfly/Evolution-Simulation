import cv2
import numpy as np
import os
import json 

from distinct_color_generator import distinct_colors
from utils import get_newest_file
import sys


os.chdir('src/logs/')

#mask = create_circular_mask((40,40), 0.05) * 0.5


img_array = []


# path = "run_2022-09-19 20 19 42"
path = get_newest_file()
# gen = 61
logged_gen = -1
if len(sys.argv) > 1:
    logged_gen = int(sys.argv[1])

gen = os.listdir(f"{path}/generations/")[logged_gen]

with open(path + '/world_configurations/world_params.json', 'r') as world_configs_file:
    world_configs = json.load(world_configs_file)
try:
    n_species = world_configs["n_species"] #json.load(path + '/world_configurations/world_params.json' )
except KeyError:
    n_species = 1

multi_mask = n_species > 1
if n_species <= 5:
    color_species = np.array([[204,  81,  81],
                            [127,  51,  51],
                            [ 81, 204, 204],
                            [ 51, 127, 127],
                            [142, 204,  81]], dtype=np.uint8)[:n_species, :]
else:
    color_species = distinct_colors(n_colors = n_species).astype(np.uint8)

mask = np.loadtxt(path + '/world_configurations/death_mask') 
world_shape = mask.shape 
mask *= 0.2                

if  os.path.exists(path + f'/world_configurations/wall_mask'):
    wall_mask = np.loadtxt(path + '/world_configurations/wall_mask')
    wall_mask *= 0.6              

else:            
    wall_mask = 0 

if not multi_mask:
    for i, file in enumerate(os.listdir(f"{path}/generations/{gen}/step_world_state")):
        img = np.loadtxt(f"{path}/generations/{gen}/step_world_state/step{i + 1}")
        img_array.append(img)
        
        height, width = img.shape
        size = (height, width)

        img = img + mask + wall_mask        
        img_resized = cv2.resize(img, (600, 600))
        cv2.imshow(f"{gen}", img_resized)
        cv2.waitKey(0)    
        
    cv2.destroyAllWindows()           
    
else:
    
    for i, file in  enumerate(os.listdir(f'{path}/generations/{gen}/step_world_state_species')):
        world_state = np.loadtxt(file).astype(np.bool_)
        
        colors_4D = np.tile(color_species.reshape(-1, 1, 3).reshape(-1, 3, 1, 1), (1, 1, *world_shape))
        world_state_not_4D = np.tile(np.logical_not(world_state).reshape(-1, 1, *world_shape),(1, 3, 1, 1))

        a = colors_4D[world_state_not_4D] = 0
        img_3D = np.sum(a, axis=0).astype(np.uint8)

        img_3D_resized = cv2.resize(img_3D, (600, 600))
        cv2.imshow(f'{gen}', img_3D_resized)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
        

