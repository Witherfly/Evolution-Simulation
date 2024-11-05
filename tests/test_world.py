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

def state_from_pop_pos(pop_pos, world_shape):
    world_state = np.zeros(world_shape)
    world_state[pop_pos[:, 0], pop_pos[:, 1]] = 1

    return world_state


params = [(np.array([[0, 0]]), [[0, 0, 0, 0, 0, 0]]), 
          (np.array([[9, 9]]), [[1, 1, 0, 0, 0, 0]]),
          (np.array([[0, 9], [1, 9]]), [[0, 1, 0, 1, 0, 0], [0.111, 1, 1, 0, 0, 0]]),
          (np.array([[3, 3], [3, 4]]), [[0.333, 0.333, 0, 0, 0, 1], [0.333, 0.444, 0, 0, 1, 0]])]
          
@pytest.mark.parametrize("pop_pos, observations", params)
def test_create_observation(world_params, pop_pos, observations):

    n_population = len(pop_pos)
    world_params['n_population'] = n_population
    world = World(**world_params)

    world.dot_objects = [Dot(i, '000') for i in range(n_population)]

    world.pop_pos = pop_pos
    world.world_state = state_from_pop_pos(pop_pos, world.world_shape)
    

    for i in range(n_population):
        obs = world.create_observation(i)
        np.testing.assert_allclose(obs, observations[i], atol=0.01)

params = [((0, 0), 0), 
          ((2, 3), 1),
          ((9, 9), 2), 
          ((1, 1), None),
          ((0, 1), None),
          ((100, 100), None),
          ((-1, -1), None),
          ((4, 6), ValueError)]
@pytest.mark.parametrize('pos, expected', params)
def test_dot_at_pos_idx(world_params, pos, expected):

    pop_pos = np.array([[0, 0], [2, 3], [9, 9], [4, 6], [4, 6]])

    world = World(**world_params)

    world.pop_pos = pop_pos
    world.n_population = len(pop_pos)
    world.world_state = state_from_pop_pos(pop_pos, world.world_shape)
    world.dot_objects = [Dot(i, '') for i in range(world.n_population)]

    if expected is None:
        assert world.dot_at_pos(pos, check_occ=True) is None
    elif isinstance(expected, int):
        assert world.dot_at_pos(pos, check_occ=True).id == expected
    else:
        with pytest.raises(expected):
            world.dot_at_pos(pos, check_occ=True)




        