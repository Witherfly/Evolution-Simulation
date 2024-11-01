import pytest 
from collections import namedtuple

from src.callbacks_module import Callback, CallbackList

@pytest.fixture
def sample_Callback() -> Callback:

    methods = dict()
    for cbk_point in Callback.callback_points:
        methods[cbk_point] = lambda *args, **kwargs: None


    callback = type("SampleCallback", (Callback,), methods)

    return callback


@pytest.fixture
def sample_CallbackList() -> CallbackList:

    n_callbacks = 2

    methods = dict()
    for cbk_point in Callback.callback_points:
        methods[cbk_point] = lambda *args, **kwargs: None

    Callback_types = [type(f"Callback{i}", (Callback,), methods) for i in range(n_callbacks)]
    callbacks = [cbk() for cbk in Callback_types]
    

    return CallbackList(callbacks)

        

class TestCallbackList:

    def test_callback_points(self, sample_CallbackList):

        World = namedtuple("World", ["create_logs", "current_gen"])
        world = World(create_logs=True, current_gen=42)
        
        for cbk_point in Callback.callback_points:

            getattr(sample_CallbackList, cbk_point)()