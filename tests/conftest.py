import pytest 
import numpy as np

@pytest.fixture
def random_seed():

    np.random.seed(42)