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




def world_framed(path_run, gen : int):


    with open(path_run + '/world_configurations/world_params.json', 'r') as world_configs_file:
        world_configs = json.load(world_configs_file)

    n_species = world_configs['n_species']
    world_shape = world_configs['world_shape']

    if n_species <= 5:
        color_species = colors_rgb
    else:
        color_species = distinct_colors(n_colors = n_species).astype(np.float32)

    zone_mask = np.loadtxt(path_run + '/world_configurations/death_mask', dtype=np.bool_)
    world_shape = zone_mask.shape 

    if  os.path.exists(path_run + f'/world_configurations/wall_mask'):
        wall_mask = np.loadtxt(path_run + '/world_configurations/wall_mask', dtype=np.bool_)

    else:            
        wall_mask = np.zeros_like(world_shape) 

    
    path_gen = get_path_gen(gen, path_run=path_run)
    stats = pd.read_csv(os.path.join(path_gen, "statistics/stats.csv"))

    path = os.path.join(path_gen, "step_pop_pos_species")
    text_space = 1.0
    plot_images = []
    plot_names = [name for name in ['total_killed', 'total_in_zone'] if name in stats.columns]
    n_steps = world_configs['n_steps']

    plot_images = plot_framed(np.arange(n_steps+1), 
                              [stats[name] for name in plot_names], 
                              Y_labels=plot_names)

    canvas_images = []

    alive_last_step = np.ones(world_configs['n_population'])
    pop_pos_last_step = np.zeros((world_configs['n_population'], 3), dtype=np.int16)
    for i, file in  enumerate(os.listdir(path)):
        print(i)
        pop_pos = np.loadtxt(os.path.join(path, f'step{i}'), dtype=np.int16)

        alive_current_step = np.all(pop_pos[:, :2] != np.array([-2, -2]), axis=1)
        killed_pos = pop_pos_last_step[np.logical_and(alive_last_step, ~alive_current_step)]
        pop_pos_alive = pop_pos[alive_current_step]

        world_3d = np.zeros((*world_shape, 3), dtype=np.float32) 
        world_3d[zone_mask, :] = np.ones((3,)) * 0.2
        world_3d[wall_mask, :] = np.ones((3,)) * 0.6
        for species in range(1, n_species+1):
            species_pos = pop_pos_alive[pop_pos_alive[:, -1] == species, :2]
            species_killed_pos = killed_pos[killed_pos[:, -1] == species, :2]
            world_3d[species_pos[:, 0], species_pos[:, 1], :] = color_species[species-1]
            world_3d[species_killed_pos[:, 0], species_killed_pos[:, 1], :] = color_species[species-1] * 0.4

        pop_pos_last_step = pop_pos.copy()
        alive_last_step = alive_current_step.copy()


        world_img_size = 1000
        img_3d_resized = cv2.resize(world_3d, (world_img_size, world_img_size))

        text_canvas = np.ones((img_3d_resized.shape[0], 
                            int(img_3d_resized.shape[1] * text_space), 
                            img_3d_resized.shape[2]), dtype=np.float32) * 0.8

        step_stats = stats.iloc[i]
        offset = 30
        cv2.putText(text_canvas, f"Generation: {gen}", 
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

    frames = np.array(canvas_images)
    return frames

def get_path_run():


    if len(sys.argv) > 2:
        run = int(sys.argv[2])
        path_run = os.listdir()[run]
    else:
        path_run = get_newest_file()
    
    return path_run

def get_gen(path_run) -> int:

    if len(sys.argv) > 1:
        logged_gen = int(sys.argv[1])
    else: 
        logged_gen = -1

    gen = sorted_alphanumeric(os.listdir(f"{path_run}/generations/"))[logged_gen]

    return int(gen[3:])

def get_path_gen(gen, path_run):
    return f'{path_run}/generations/gen{gen}'


def load_frames(path_gen):
    frames = np.load(path_gen + '/video/frames.npz')
    return frames["arr_0"]

def save_frames(path_gen, frames):

    if os.path.exists(path_gen + '/video'):
        return 
    os.mkdir(path_gen + '/video')
    np.savez_compressed(path_gen + '/video/frames', frames)

def get_frames(return_paths=False):

    os.chdir('logs/')
    
    path_run = get_path_run()
    gen = get_gen(path_run=path_run)
    path_gen = get_path_gen(gen, path_run=path_run)

    frames_already_exists = True if os.path.exists(path_gen + '/video') else False

    if frames_already_exists:
        frames = load_frames(path_gen)
    else:
        frames = world_framed(path_run=path_run, gen=gen)
    
    if return_paths:
        return frames, {'path_run': path_run, 'path_gen': path_gen, 'gen': gen}
    
    return frames

if __name__ == '__main__':
    
    frames, info_dict = get_frames(return_paths=True)

    i = 0
    i_max = len(frames) - 1
    while True:
        
        cv2.imshow(f'gen{info_dict["gen"]}', frames[i])

        pressed_key = cv2.waitKeyEx()

        if pressed_key == 27:
            break
        if pressed_key == 2424832:
            i -= 1
        else: 
            i += 1
        
        i = i % (i_max +1)

    cv2.destroyAllWindows()

    # save_frames(info_dict["path_gen"], frames)

