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
              
    def predict(self, input : npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        
        input = np.append(input, 1.0).reshape(-1, 1)

        hidden = np.append(np.tanh( self.weight_ih @ input), 1.0).reshape(-1, 1)
        output = np.tanh(self.weight_io @ input + self.weight_ho @ hidden)
        #actions = (output >= 0.5) + (-1 * (output <= -0.5))  

        
        
        action : int = np.argmax(np.ravel(output)).item()
        action_array = np.zeros((self.output_size,), dtype=np.bool_)
        action_array[action] = True
        
        return action_array