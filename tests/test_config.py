import numpy as np 
import pytest

from src.genetic import *
from src import selection_funcs
from src import obstacles
from src import callbacks_module


#############################################

world_shape = (50, 50)
n_max_gen = 1000
n_logs = 15  

death_func1 = selection_funcs.Zone_selection("east_west", 0.1, world_shape)
death_func2 = selection_funcs.Zone_selection("circle", 0.1, world_shape)
death_func3 = selection_funcs.Zone_selection(("circle", "east_west"), (0.15, 0.06), world_shape)
death_func4 = selection_funcs.Zone_selection(("circle", "east_west", "north_south"), (0.15, 0.1, 0.1), world_shape)
 
death_func_custom = selection_funcs.Load_custom("death_masks", "newest")

wall_mask1 = obstacles.Wall(world_shape).mask

wall_mask_custom = obstacles.Load_wall("wall_masks", "newest").mask 

# callbacks 
world_state_logging = callbacks_module.LogWorldState('polynomial', n_max_gen, n_logs=n_logs, log_pos=False)
weights_logging = callbacks_module.LogWeights('polynomial', n_max_gen, n_logs=n_logs)
time_gens = callbacks_module.TimeGens()


world = World(world_shape=world_shape, n_population=150, n_steps=100, n_max_gen=n_max_gen,
              mutation_rate=0.3, flip_rate=0.2, n_connections=10, create_logs=False,
              kill_enabled=False, wall_mask=wall_mask_custom, death_func=death_func_custom,
              no_spawn_in_zone=True, crossover_func_name="one_point_crossover",
              callbacks=[world_state_logging, time_gens])

##################################################

def test_config():

    assert set(world.get_config()).issubset(set(world.__dict__))

