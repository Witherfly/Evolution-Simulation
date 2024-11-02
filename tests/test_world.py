import pytest 

from src.world import World 
from src.dot import Dot
from src.selection_funcs import Zone_selection

from typing import Any, Callable
from collections import namedtuple

import numpy as np
import numpy.typing as npt



@pytest.fixture
def world_params() -> dict:
    world_shape = (10, 10)
    death_func = Zone_selection("east_west", 0.2, world_shape)

    world_params = {"world_shape":world_shape, 
            "n_population":10,
            "n_steps":20,
            "n_max_gen":100,
            "n_connections":12,
            "create_logs":False,
            "death_func":death_func}

    return world_params

params = [(np.array([[0, 0]]), [[0, 0, 0, 0, 0, 0]]), 
          (np.array([[9, 9]]), [[1, 1, 0, 0, 0, 0]]),
          (np.array([[0, 9], [1, 9]]), [[0, 1, 0, 1, 0, 0], [0.111, 1, 1, 0, 0, 0]]),
          (np.array([[3, 3], [3, 4]]), [[0.333, 0.333, 0, 0, 0, 1], [0.333, 0.444, 0, 0, 1, 0]])]
          
@pytest.mark.parametrize("world_pop_pos, observations", params)
def test_create_observation(world_params, world_pop_pos, observations):

    n_population = len(world_pop_pos)
    world_params['n_population'] = n_population
    world = World(**world_params)

    world.dot_objects = [Dot(i, '000') for i in range(n_population)]

    world.pop_pos = world_pop_pos
    # np.array([[0, 0],
    #                         [world.world_shape[0] - 1, world.world_shape[1] - 1],
    #                         [3, 3],
    #                         [3, 4],
    #                         [0, 9],
    #                         [1, 9]])
    world.world_state = np.zeros(world.world_shape)
    world.world_state[world.pop_pos[:, 0], world.pop_pos[:, 1]] = 1
    
    # observations = [[0, 0, 0, 0, 0, 0],
    #                 [1, 1, 0, 0, 0, 0],
    #                 [0.333, 0.333, 0, 1, 0, 0],
    #                 [0.333, 0.444, 1, 0, 0, 0],
    #                 [0, 1, 0, 0, 0, 1],
    #                 [0.111, 1, 0, 0, 1, 0]]

    for i in range(n_population):
        obs = world.create_observation(i)
        np.testing.assert_allclose(obs, observations[i], atol=0.01)
        