import numpy as np 
import numpy.typing as npt

class Brain():

    
    def __init__(self, neuron_pairs : list[npt.NDArray[np.int16] | None], 
                 pair_weights : list[float] , input_size : int, hidden_size  : int , output_size : int):
        
        self.neuron_pairs = neuron_pairs # [[3, 0, 1], [0, 2, 4]]
        self.pair_weights = pair_weights # [-3.6, 2.3, 0.1]
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # self.hidden_bias = np.zeros((3, 1))
        # self.hidden_bias[1, 0] = 1
        # self.hidden_bias[2, 0] = -1
        
    def build_nn(self):
        
        self.weight_ih = np.zeros((self.hidden_size, self.input_size+1))
        self.weight_io = np.zeros((self.output_size, self.input_size+1))
        self.weight_ho = np.zeros((self.output_size, self.hidden_size+1))
        
        
        
        for pair, weight in zip(self.neuron_pairs, self.pair_weights):
            
            if pair is not None:
                if pair[2] == -1000:
                    self.weight_ih[pair[1], pair[0]] = weight
                    
                elif pair[1] == -1000:
                    self.weight_io[pair[2], pair[0]] = weight 
                    
                elif pair[0] == -1000:
                    self.weight_ho[pair[2], pair[1]] = weight 
                    
        self.all_weights = [self.weight_ih, self.weight_io, self.weight_ho]
              
    def predict(self, input : npt.NDArray[np.float32]) -> int:
        
        input = np.append(input, 1.0).reshape(-1, 1)

        hidden = np.append(np.tanh( self.weight_ih @ input), 1.0).reshape(-1, 1)
        output = np.tanh(self.weight_io @ input + self.weight_ho @ hidden)
        #actions = (output >= 0.5) + (-1 * (output <= -0.5))  

        
        
        action : int = np.argmax(np.ravel(output)).item()
        
        return action





class Brain2:

    def __init__(self, neuron_pairs, n_neurons_per_layer):

        self.neuron_pairs = neuron_pairs
        self.n_neurons_per_layer = n_neurons_per_layer

    @staticmethod
    def init_nn(n_neurons_per_layer):
        all_weights = []
        n_layers = len(n_neurons_per_layer)
        for i in range(n_layers-1):
            
            # self.all_weights.append(np.empty((self.n_neurons_per_layer[i+1], self.n_neurons_per_layer[i])))
            for j in range(i+1, n_layers):
                all_weights.append(np.empty((n_neurons_per_layer[j], n_neurons_per_layer[i])))

        return all_weights

    @staticmethod
    def build_nn(neuron_pairs, all_weights):

        for i, layer in enumerate(neuron_pairs):
            
            all_weights[i][layer[:, 1].astype(int), layer[:, 0].astype(int)] = layer[:, 2]

        return all_weights

class SkipNeuralNet:

    def __init__(self, all_weights, n_neurons_per_layer, activation_func=np.tanh):           
        self.all_weights = all_weights
        self.output_size = self.all_weights[-1].shape[0]
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation_func = activation_func

    def get_configs(self):
        return {key:val for key, val in self.__dict__.items() if key in ['n_neurons_per_layer', 'activation_func']}

    @classmethod
    def random_init(cls, n_neurons_per_layer):

        rng = np.random.default_rng()
        all_weights = []
        n_layers = len(n_neurons_per_layer)
        for i in range(n_layers-1):
            
            # self.all_weights.append(np.empty((self.n_neurons_per_layer[i+1], self.n_neurons_per_layer[i])))
            for j in range(i+1, n_layers):
                weights_ij = rng.normal(size=(n_neurons_per_layer[j], n_neurons_per_layer[i]+1)) # 2 ** 15 = 32768
                all_weights.append(weights_ij) 
            
        return cls(all_weights, n_neurons_per_layer)


    def predict(self, input) -> int:

        n_layers = len(self.n_neurons_per_layer)
        layer_outputs = np.empty((len(self.n_neurons_per_layer),
                                   max(self.n_neurons_per_layer)+1))
        layer_outputs[0, 1:self.n_neurons_per_layer[0]+1] = input
        layer_outputs[:, 0] = 1
        
            
        for i in range(1, n_layers):

            idx = i - 1

            output = layer_outputs[i, 1:self.n_neurons_per_layer[i]+1].copy()
            k = 0
            for j in range(n_layers-2, n_layers-i, -1):
                output += self.all_weights[idx] @ layer_outputs[k, :self.n_neurons_per_layer[k]+1]
                idx += j
                k += 1
            
            layer_outputs[i, 1:self.n_neurons_per_layer[i]+1] = self.activation_func(output)

        
        output = layer_outputs[-1, 1:self.n_neurons_per_layer[-1]+1]
        action : int = np.argmax(np.ravel(output)).item()
        
        return action






