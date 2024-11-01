import numpy as np
import numpy.typing as npt
from .brain import Brain

class Dot():
    
    def __init__(self, id : int, genome : str, species : int | None = None):
        
        self.id = id
        self.genome = genome 
        self.alive = True 
        self.species = species
    
    @staticmethod
    def create_genome(n_connections, n_dif_inputs, n_dif_hidden, n_dif_outputs) -> str:
        
        genome = ''
                
        for i in range(n_connections):
            
            is_hidden =  np.random.choice([0, 1], p=[0.7, 0.3])
            is_output = np.random.choice([0, 1], p=[0.7, 0.3])
            
            if  is_hidden == 0:
                n1 = np.random.randint(n_dif_inputs)
            else:
                n1 = np.random.randint(n_dif_hidden)
            
            if is_output == 1:
                n2 = np.random.randint(n_dif_outputs)
            else:
                n2 = np.random.randint(n_dif_hidden)
                
            weight = np.random.randint(-32768, 32768) # 2 ** 15 = 32768
            
            
            
            n1_id_bin = str(is_hidden) + format(n1, '05b')
            n2_id_bin = str(is_output) + format(n2, '05b')
            
            sign_bit = np.sign(weight) if np.sign(weight) == 1 else 0
            weight_bin =  str(sign_bit) + format(abs(weight), '015b')
            
            gen = n1_id_bin + n2_id_bin + weight_bin
        
            genome += gen + 'x' #marks end of each gene
        
        return genome

       
        
    def unencode_genome(self, n_dif_input : int, n_dif_hidden : int, n_dif_output : int):
         
        genes : list[str] = self.genome.split('x')[:-1]
        neuron_pairs  = []
        pair_weights : list[float] = []

        for gene in genes:

            neuron_array = np.empty((3,), np.int16)
            
            n1 = int(gene[1:6], 2)
            
            is_hidden = int(gene[0])
            is_output = int(gene[6])
            
            if is_hidden == 0:
                
                if n1 >= n_dif_input:
                    n1 = int(n1 % n_dif_input)
                neuron_array[0] = n1
                neuron_array[1] = -1000
                
            elif is_hidden == 1:
                
                if n1 >= n_dif_hidden:
                    n1 = int(n1 % n_dif_hidden)
                neuron_array[1] = n1
                neuron_array[0] = -1000
            
            n2 = int(gene[7:12], 2)
            if is_output == 1:
                
                if n2 >= n_dif_output:
                    n2 = int(n2 % n_dif_output)
                    
                neuron_array[2] = n2
                if not is_hidden:
                    neuron_array[1] = -1000
                
            
            elif is_output == 0 and is_hidden == 0:
                
                if n2 >= n_dif_hidden:
                    n2 = int(n2 % n_dif_hidden) 
                    
                neuron_array[2] = -1000
                neuron_array[1] = n2 
                
                
            else:
                neuron_array = None 
                
                
            weight = gene[12:]
            sign = 1 if weight[0] == '1' else -1

            weight = sign * int(weight[1:], 2) / 8000

            neuron_pairs.append( neuron_array )
            pair_weights.append( weight )         

         
        
        self.brain = Brain(neuron_pairs, pair_weights, n_dif_input, n_dif_hidden, n_dif_output)
        self.brain.build_nn()
            
    def move(self, inputs : npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        
        action = self.brain.predict(inputs)
        
        return action