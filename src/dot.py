import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from brain import Brain, SkipNeuralNet

class Dot:
    def __init__(self, id : int | None = None, species : int | None = None):
        
        self.id = id
        self.species = species
        self.alive = True 
        self.brain : Brain = None

    @abstractmethod
    def random_init(self, n_connections, n_neurons_per_layer):
        pass

    @abstractmethod
    def unencode_genome(self, *args, **kwargs) -> None:
        pass

    def move(self, inputs : npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        
        action = self.brain.predict(inputs)
        
        return action

class DotGeneless(Dot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def random_init(self, n_connections, n_neurons_per_layer: tuple[int], brain_type=SkipNeuralNet) -> None:

        self.brain = brain_type.random_init(n_neurons_per_layer)

    def unencode_genome(self, *args, **kwargs):
        pass 

class DotGenetic(Dot):
    
    def __init__(self, *args, genome : str | None = None, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.genome : str = genome 
    
    
    def random_init(self, n_connections, n_neurons_per_layer) -> None:
        
        genome = ''
        n_inputs, n_hidden, n_outputs = n_neurons_per_layer
                
        for i in range(n_connections):
            
            is_hidden =  np.random.choice([0, 1], p=[0.7, 0.3])
            is_output = np.random.choice([0, 1], p=[0.7, 0.3])
            
            if  is_hidden == 0:
                n1 = np.random.randint(n_inputs)
            else:
                n1 = np.random.randint(n_hidden)
            
            if is_output == 1:
                n2 = np.random.randint(n_outputs)
            else:
                n2 = np.random.randint(n_hidden)
                
            weight = np.random.randint(-32768, 32768) # 2 ** 15 = 32768
            
            
            
            n1_id_bin = str(is_hidden) + format(n1, '05b')
            n2_id_bin = str(is_output) + format(n2, '05b')
            
            sign_bit = np.sign(weight) if np.sign(weight) == 1 else 0
            weight_bin =  str(sign_bit) + format(abs(weight), '015b')
            
            gen = n1_id_bin + n2_id_bin + weight_bin
        
            genome += gen + 'x' #marks end of each gene
        
        self.genome = genome

        self.unencode_genome(n_neurons_per_layer)

          
    def unencode_genome(self, n_neurons_per_layer : tuple[int, ...]):
         
        n_input, n_hidden, n_output = n_neurons_per_layer
        genes : list[str] = self.genome.split('x')[:-1]
        neuron_pairs  = []
        pair_weights : list[float] = []

        for gene in genes:

            neuron_array = np.empty((3,), np.int16)
            
            n1 = int(gene[1:6], 2)
            
            is_hidden = int(gene[0])
            is_output = int(gene[6])
            
            if is_hidden == 0:
                
                if n1 >= n_input:
                    n1 = int(n1 % n_input)
                neuron_array[0] = n1
                neuron_array[1] = -1000
                
            elif is_hidden == 1:
                
                if n1 >= n_hidden:
                    n1 = int(n1 % n_hidden)
                neuron_array[1] = n1
                neuron_array[0] = -1000
            
            n2 = int(gene[7:12], 2)
            if is_output == 1:
                
                if n2 >= n_output:
                    n2 = int(n2 % n_output)
                    
                neuron_array[2] = n2
                if not is_hidden:
                    neuron_array[1] = -1000
                
            
            elif is_output == 0 and is_hidden == 0:
                
                if n2 >= n_hidden:
                    n2 = int(n2 % n_hidden) 
                    
                neuron_array[2] = -1000
                neuron_array[1] = n2 
                
                
            else:
                neuron_array = None 
                
                
            weight = gene[12:]
            sign = 1 if weight[0] == '1' else -1

            weight = sign * int(weight[1:], 2) / 8000

            neuron_pairs.append( neuron_array )
            pair_weights.append( weight )         

         
        
        self.brain = Brain(neuron_pairs, pair_weights, n_input, n_hidden, n_output)
        self.brain.build_nn()
            