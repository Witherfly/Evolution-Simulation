import pytest 
from src.custom_callbacks import InitLogs, LogWorldState

from dataclasses import dataclass
from collections import namedtuple
import numpy as np
import os
import json
from unittest import mock
import numpy.typing as npt
from typing import Callable, Any


@dataclass 
class MockWorld:
    world_shape : tuple
    death_func : Any
    wall_mask : npt.NDArray
    obstacles_enabled : bool
    get_config : Callable

@pytest.fixture
def sample_world() -> MockWorld:


    # World = namedtuple('World', ['world_shape', 'death_func', 'wall_mask', 'obstacles_enabled', 'get_config'])
    Death_func = namedtuple('Death_func', ['mask'])
    world_shape = (50, 50)
    
    death_mask = np.random.randint(0, 2, size=world_shape)
    wall_mask = np.random.randint(0, 2, size=world_shape)

    death_func = Death_func(mask=death_mask)

    world = MockWorld(world_shape, death_func, wall_mask, obstacles_enabled=True, get_config=lambda *args : None)

    return world

FAKE_TIME_STR = "2021-09-01T00:00:00.000000"

@pytest.fixture
def patch_datetime_now():
    with mock.patch("src.custom_callbacks.datetime") as mock_datetime:
        mock_datetime.now.return_value.isoformat.return_value = FAKE_TIME_STR
        yield mock_datetime


def test_InitLogs(sample_world, tmp_path, mocker):

    os.mkdir(tmp_path / "src/")
    os.mkdir(tmp_path / "src/logs")
    
    config_dict = {"var1" : "val1", "var2" : 2, "var3" : True}
    #sample_world.get_config = lambda *args: config_dict

    fake_now_formatted = "2000-01-13 01 02 03"
    # mocker.patch("src.custom_callbacks.datetime.now", return_value=mock_datetime)

    mocker.patch.object(sample_world, "get_config", return_value=config_dict)
    #mocker.patch("sample_world.get_config", return_value=config_dict)

    init_logs = InitLogs()
    init_logs.on_init_simulation(sample_world)

    # assert os.getcwd() == tmp_path /"src/logs" / f"run_{fake_now_formatted}"

    # assert os.path.exists(tmp_path / "src/logs/" / f"run_{fake_now_formatted}")

    assert os.path.exists("world_configurations")
    assert os.path.exists("generations")
    assert os.path.exists("colorizing_data")
    assert os.path.exists("performance")
    assert os.path.exists("world_configurations/world_params.json")

    with open("world_configurations/world_params.json", "r") as f:
        assert json.load(f) == config_dict

    death_mask = np.loadtxt("world_configurations/death_mask")
    np.testing.assert_equal(death_mask, sample_world.death_func.mask)

    wall_mask = np.loadtxt("world_configurations/wall_mask")
    np.testing.assert_equal(wall_mask, sample_world.wall_mask)

def test_LogWorldState(sample_world, tmp_path):

    os.chdir(tmp_path)
    os.mkdir(tmp_path / "generations")

    sample_world.current_gen = 42
    sample_world.current_step = 1
    sample_world.n_species = 1
    mock_world_state = np.random.randint(0, 2, size=sample_world.world_shape)
    sample_world.world_state = mock_world_state

    cbk = LogWorldState(log_points=[42], n_max_gen=100, n_logs=1)
    cbk.on_gen_begin(sample_world)
    
    cbk.on_step_end(sample_world)

    world_state = np.loadtxt(f"generations/gen{sample_world.current_gen}/step_world_state/step{sample_world.current_step}")
    np.testing.assert_equal(world_state, mock_world_state)

