import cv2
import numpy as np
import os
import json 
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from itertools import repeat
from functools import partial

from distinct_color_generator import distinct_colors
from utils import get_newest_file, text_specs, sorted_alphanumeric
import sys
from utils import colors_rgb
from frame_plot import plot_framed




def world_framed():
    os.chdir('src/logs/')

    # path = "run_2022-09-19 20 19 42"
    path = get_newest_file()
    # gen = 61
    logged_gen = -1
    if len(sys.argv) > 1:
        logged_gen = int(sys.argv[1])

    if len(sys.argv) > 2:
        run = int(sys.argv[2])
        path = os.listdir()[run]

    gen = sorted_alphanumeric(os.listdir(f"{path}/generations/"))[logged_gen]

    with open(path + '/world_configurations/world_params.json', 'r') as world_configs_file:
        world_configs = json.load(world_configs_file)

    n_species = world_configs['n_species']
    world_shape = world_configs['world_shape']

    if n_species <= 5:
        color_species = colors_rgb
    else:
        color_species = distinct_colors(n_colors = n_species).astype(np.float32)

    zone_mask = np.loadtxt(path + '/world_configurations/death_mask', dtype=np.bool_)
    world_shape = zone_mask.shape 

    if  os.path.exists(path + f'/world_configurations/wall_mask'):
        wall_mask = np.loadtxt(path + '/world_configurations/wall_mask', dtype=np.bool_)

    else:            
        wall_mask = np.zeros_like(world_shape) 

    path = f'{path}/generations/{gen}'
    
    stats = pd.read_csv(os.path.join(path, "statistics/stats.csv"))

    path = os.path.join(path, "step_pop_pos_species")
    text_space = 1.0
    plot_images = []
    plot_names = [name for name in ['total_killed', 'total_in_zone'] if name in stats.columns]
    n_steps = world_configs['n_steps']

    plot_images = plot_framed(np.arange(1, n_steps+1), 
                              [stats[name] for name in plot_names], 
                              Y_labels=plot_names)

    canvas_images = []

    for i, file in  enumerate(os.listdir(path)):
        print(i)
        pop_pos = np.loadtxt(os.path.join(path, f'step{i+1}'), dtype=np.int16)

        world_3d = np.zeros((*world_shape, 3), dtype=np.float32) 
        world_3d[zone_mask, :] = np.ones((3,)) * 0.2
        world_3d[wall_mask, :] = np.ones((3,)) * 0.6
        for species in range(1, n_species+1):
            species_pos = pop_pos[pop_pos[:, -1] == species, :2]
            world_3d[species_pos[:, 0], species_pos[:, 1], :] = color_species[species-1]

        img_3d_resized = cv2.resize(world_3d, (1000, 1000))
        text_canvas = np.ones((img_3d_resized.shape[0], 
                            int(img_3d_resized.shape[1] * text_space), 
                            img_3d_resized.shape[2]), dtype=np.float32) * 0.8

        step_stats = stats.iloc[i]
        offset = 30
        cv2.putText(text_canvas, f"Generation: {gen[3:]}", 
                    org=(20, offset), **text_specs)
        
        text_names = [name for name in stats.columns if name not in plot_names]
        for col_idx, col_name in enumerate(text_names):
            offset += 30
            text_specs_copy = text_specs.copy()
            # if 'species' in col_name:
            #     text_specs_copy = {**text_specs, 'color' : tuple((color_species[int(col_name[-1])]*255).astype(int))}

            cv2.putText(text_canvas, f"{col_name}".replace('_', ' ') + f": {step_stats[col_name]}", 
                        org=(20, offset), **text_specs_copy)
        
        plot_img = plot_images[i]
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)
        plot_img = cv2.resize(plot_img, (text_canvas.shape[0] , int(text_canvas.shape[1]*0.7))).astype(np.float32) / 255
        # text_canvas[plot_img.shape[0]:, :, :] *= 0
        # text_canvas[plot_img.shape[0]:, :, :][plot_img > 0.001] *= 0
        text_canvas[-plot_img.shape[0]:, :, :] = plot_img

        canvas = np.concatenate((img_3d_resized, text_canvas), axis=1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) 

        canvas_images.append(canvas)

    return canvas_images
        



if __name__ == '__main__':
    
    frames = world_framed()
    i = 0
    i_max = len(frames) - 1
    while True:
        # if i < 0:
        #     i = i_max 
        
        cv2.imshow(f'gen', frames[i])

        if cv2.waitKeyEx() == 2424832:
            i -= 1
        else: 
            i += 1

        if i > i_max:
            break
    
    cv2.destroyAllWindows()

