import pytest 
import numpy as np

from src.dot import Dot 

np.random.seed(42)

params = [(10, 6, 3, 4), (10, 10, 10, 10), (1, 1, 1, 1), (10, 1, 1, 1)]

@pytest.mark.parametrize("n_connections, n_dif_inputs, n_dif_hidden, n_dif_outputs", params)
def test_genome_coding(n_connections, n_dif_inputs, n_dif_hidden, n_dif_outputs):

    genome = Dot.create_genome(n_connections, n_dif_inputs, n_dif_hidden, n_dif_outputs)

    d = Dot(1, genome)

    d.unencode_genome(n_dif_inputs, n_dif_hidden, n_dif_outputs)

    obs_array = np.random.uniform(0, 1, size=n_dif_inputs)
    action_array = d.move(obs_array)

    assert np.count_nonzero(action_array) == 1



    
