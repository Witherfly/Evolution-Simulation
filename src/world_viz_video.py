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



def image_from_plot(step, Y, **kwargs):
    
    fig = plt.figure()
    fig.tight_layout(pad=0)
    # fig.add_subplot(111)
    ax = fig.add_axes([0, 0, 1, 1])
    x = np.arange(1, step+1)
    for y in Y:
        ax.plot(x, y)
    ax.margins(0.01)
    ax.grid()
    ax.set_xlabel('steps')
    ax.set_xlim(1, 70)
    ax.set_facecolor((0, 0, 0))
    try:
        ax.set_ylim(0, kwargs["ylim"])
    except KeyError:
        pass
    # ax.set_ylim(0, max(x))


    return image_from_fig(fig)

def image_from_fig(fig):

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()

    return data


def main():
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

    if n_species == 1:
        previous_img = np.zeros_like(world_shape)
        for i, file in enumerate(os.listdir(f"{path}/generations/{gen}/step_world_state")):
            img = np.loadtxt(f"{path}/generations/{gen}/step_world_state/step{i + 1}")
            if np.isclose(img, previous_img):
                break
            previous_img = img.copy()
            
            height, width = img.shape
            size = (height, width)

            zone_mask *= 0.2                
            wall_mask *= 0.6              
            img = img + zone_mask + wall_mask        
            img_resized = cv2.resize(img, (600, 600))
            cv2.imshow(f"{gen}", img_resized)
            cv2.waitKey(0)    
            
        cv2.destroyAllWindows()           
        
    else:

        path = f'{path}/generations/{gen}'
        
        stats = pd.read_csv(os.path.join(path, "statistics/stats.csv"))

        path = os.path.join(path, "step_pop_pos_species")
        text_space = 1.0
        plot_images = []
        plot_names = [name for name in ['total_killed', 'total_in_zone'] if name in stats.columns]
        n_steps = world_configs['n_steps']
        # with multiprocessing.Pool() as pool:
        #     plot_images = pool.starmap(partial(image_from_plot, ylim=stats[plot_names].max(axis=None) * 1.05), 
        #                 zip(range(1, n_steps+1),
        #                 ([stats[name].iloc[:i+1] for name in plot_names] for i in range(n_steps))
        #                 ))
        for i in range(world_configs['n_steps']):
            plot_img = image_from_plot(i+1, [stats[name].iloc[:i+1] for name in plot_names], 
                                       ylim=stats[plot_names].max(axis=None) * 1.05)
            plot_images.append(plot_img)


        for i, file in  enumerate(os.listdir(path)):

            pop_pos = np.loadtxt(os.path.join(path, f'step{i+1}'), dtype=np.int16)

            world_3d = np.zeros((*world_shape, 3), dtype=np.float32) 
            world_3d[zone_mask, :] = np.ones((3,)) * 0.2
            world_3d[wall_mask, :] = np.ones((3,)) * 0.6
            for species in range(1, n_species+1):
                species_pos = pop_pos[pop_pos[:, -1] == species, :2]
                world_3d[species_pos[:, 0], species_pos[:, 1], :] = color_species[species-1]

            img_3d_resized = cv2.resize(world_3d, (600, 600))
            text_canvas = np.ones((img_3d_resized.shape[0], 
                                int(img_3d_resized.shape[1] * text_space), 
                                img_3d_resized.shape[2]), dtype=np.float32) * 0.8

            step_stats = stats.iloc[i]
            offset = 0
            for col_idx, col_name in enumerate(stats.columns):
                offset += 30
                text_specs_copy = text_specs.copy()
                # if 'species' in col_name:
                #     text_specs_copy = {**text_specs, 'color' : tuple((color_species[int(col_name[-1])]*255).astype(int))}

                cv2.putText(text_canvas, f"{col_name}".replace('_', ' ') + f": {step_stats[col_name]}", 
                            org=(20, offset), **text_specs_copy)
            
            plot_img = plot_images[i]
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)
            plot_img = cv2.resize(plot_img, (text_canvas.shape[0] , text_canvas.shape[1] // 2)).astype(np.float32) / 255
            # text_canvas[plot_img.shape[0]:, :, :] *= 0
            # text_canvas[plot_img.shape[0]:, :, :][plot_img > 0.001] *= 0
            text_canvas[plot_img.shape[0]:, :, :] += plot_img

            canvas = np.concatenate((img_3d_resized, text_canvas), axis=1)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR) 
            # img_3d_resized = world_3d


            # cv2.imshow('plot', plot_img)
            cv2.imshow(f'{gen}', canvas)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        



if __name__ == '__main__':
    main()

