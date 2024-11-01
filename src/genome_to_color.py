import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Genome_to_color():
    
    def __init__(self, color_mode, n_dif_input, n_dif_output, n_connections):
        
        self.n_dif_input = n_dif_input
        self.n_dif_ouput = n_dif_output
        self.n_connections = n_connections
        
        #self.is_first_time = True
        
        self.scaler = StandardScaler()
        self.scaler2 = StandardScaler()
        
        if color_mode == "rgb":
            self.color_dim = 3
        else:
            self.color_dim = 1
            
        self.pca = PCA(n_components=self.color_dim)
        
    
    
    def fit(self, genome_array):
        genome_array_scaled = self.scaler.fit_transform(genome_array)
        genome_array_transformed = self.pca.fit_transform(genome_array_scaled)
        self.scaler2.fit(genome_array_transformed)
        
    def transform(self, genome_array):
        
        genome_scaled = self.scaler.transform(genome_array)
            
        genome_reduced = self.pca.transform(genome_scaled)
        
        #genome_reduced_scaled = self.scaler2.transform(genome_reduced) * 73.6121593 + 122 #std of uniform =  255/(2*sqrt(3))
        
        return genome_reduced
    
        
        
"""      
   
   def reduce_to_color(self, genome_array):
        

        if self.is_first_time:
            genome_array_scaled = self.scaler.fit_transform(genome_array)
            self.pca.fit(genome_array_scaled)
            self.is_first_time = False 
        else:
            genome_array_scaled = self.scaler.transform(genome_array)
            
        genome_reduced = self.pca.transform(genome_array_scaled)
            
        return genome_reduced 
    
    def unencode_reduce(self, genome_list):
        
        n_population = len(genome_list)
        
        
        genomes_dec_array = np.empty((n_population, self.n_connections * 3))
        
        for i, genome in enumerate(genome_list):
                  
            genome_dec = self.unencode_genome(genome) #([n1, n2, n1, n2], [-3.4, 3., 0.1, -1.9])
            
            genomes_dec_array[i, self.n_connections * 2:] = np.array(genome_dec[0])
            genomes_dec_array[i, :self.n_connections * 2] = np.array(genome_dec[1])
            
        genome_array_reduced = self.reduce_to_color(genomes_dec_array)
        
        return genome_array_reduced
        
 """