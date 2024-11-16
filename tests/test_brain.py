import pytest 
import numpy as np

from src.brain import Brain2, SkipNeuralNet

def gauss(n) -> int:
    return int(n * (n + 1) / 2)

class TestBrain2:

    @pytest.mark.parametrize('n_neurons_per_layer', [(1, 2), (1, 2, 3), (1, 2, 3, 4, 5, 6)])
    def test_init_nn(self, n_neurons_per_layer):
        
        n_layers = len(n_neurons_per_layer)

        all_weights = Brain2.init_nn(n_neurons_per_layer)

        assert len(all_weights) == gauss(n_layers-1)

        for i in range(n_layers):
            
            for j in range(1, n_layers-i):
                assert all_weights[gauss(i)+j].shape == (n_neurons_per_layer[i+j], n_neurons_per_layer[i])

    def test_build_nn(self):

        all_weights = Brain2.init_nn((2, 3, 2))

        neuron_pairs = [np.array([[1, 0, 0.1],
                                  [1, 2, 0.2]]),
                        np.array([[0, 1, 0.1], 
                                  [1, 1, 0.2]]),
                        np.array([[2, 1, 0.1]])]
        
        all_weights_gold = [np.array([[0, 0.1],
                                      [0, 0],
                                      [0, 0.2]]), 
                            np.array([[0, 0],
                                      [0.1, 0.2]]),
                            np.array([[0, 0, 0],
                                      [0, 0, 0.1],])]

        all_weights = Brain2.build_nn(neuron_pairs, all_weights)

        for i in range(len(all_weights)):

            np.testing.assert_almost_equal(all_weights[i], all_weights_gold[i])



class TestSkipNeuralNet:

    @pytest.mark.parametrize('n_neurons_per_layer', [(1, 2), (1, 2, 3), (1, 2, 3, 4, 5, 6)])
    def test_init_nn(self, n_neurons_per_layer):
        
        n_layers = len(n_neurons_per_layer)

        nn = SkipNeuralNet.random_init(n_neurons_per_layer)

        assert len(nn.all_weights) == gauss(n_layers-1)


        idx = 0
        for i in range(n_layers-1):

            for j in range(i+1, n_layers):
                assert nn.all_weights[idx].shape == (n_neurons_per_layer[j], n_neurons_per_layer[i]+1)
                idx += 1

        for i in range(1, n_layers):

            idx = i - 1

            assert nn.all_weights[idx].shape == (n_neurons_per_layer[i], n_neurons_per_layer[0]+1)
            k = 1
            for j in range(n_layers-2, n_layers-i, -1):
                idx += j
                assert nn.all_weights[idx].shape == (n_neurons_per_layer[i], n_neurons_per_layer[k]+1)
                k += 1
    

    def test_predict(self):
        
        n_neurons_per_layer = (1, 2, 3, 4)
        all_weights = [np.eye(2),
                       np.eye(3, 2),
                       np.eye(4, 2),
                       np.eye(3, 3),
                       np.eye(4, 3),
                       np.eye(4, 4)]
        nn = SkipNeuralNet(all_weights=all_weights, 
                           n_neurons_per_layer=n_neurons_per_layer,
                           activation_func=lambda x : x * (x > 0))

        res = nn.predict(np.array([1]))

        assert len(res) == n_neurons_per_layer[-1]
        np.testing.assert_almost_equal(res, np.array([1, 1, 0, 0]))
        



    