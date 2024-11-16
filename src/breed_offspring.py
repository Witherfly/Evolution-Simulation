import numpy as np
import random 

from dot import Dot, DotGenetic, DotGeneless
from brain import Brain, SkipNeuralNet
from typing import Callable



    


def crossover(parent_objects : list[Dot], crossover_func : Callable, n_population : int, dot_type : Dot) -> list[Dot]:
    
    new_dot_objects = []
    
    n_parents = 2
    if len(parent_objects) < n_parents:
        raise Exception(f"only {len(parent_objects)} given, but at least expecting {n_parents}")

                
    for _ in range(n_population // n_parents + 1):
    
        parents = random.sample(parent_objects, n_parents)
        
        offspring = crossover_func(parents, dot_type)
        
        for o in offspring:
            new_dot_objects.append(o)
        
    return new_dot_objects[:n_population]
        
def one_point_crossover(parents : list[Dot], dot_type : Dot) -> list[Dot]:
    
    genome_len = len(parents[0].genome)

    idx = np.random.randint(genome_len) # x's between genes dont need to be stript
    
    head_a, tail_a = parents[0].genome[:idx], parents[0].genome[idx:]
    head_b, tail_b = parents[1].genome[:idx], parents[1].genome[idx:]
    
    offspring_a_genome = head_a + tail_b 
    offspring_b_genome = head_b + tail_a 
    
    new_offspring = []
    
    new_offspring.append(dot_type(genome=offspring_a_genome))
    new_offspring.append(dot_type(genome=offspring_b_genome))
    
    return new_offspring 

def weights_row_crossover(parents : list[Dot], dot_type=DotGeneless):
    
    rng = np.random.default_rng()
    all_weights1 = []
    all_weights2 = []
    for i in range(len(parents[0].brain.all_weights)):
        new_weight1 = np.empty_like(parents[0].brain.all_weights[i])
        new_weight2 = np.empty_like(parents[0].brain.all_weights[i])
        row_mask = rng.integers(0, 2, size=new_weight1.shape[0]).astype(np.bool_)

        new_weight1[row_mask, :] = parents[0].brain.all_weights[i][row_mask, :]
        new_weight1[~row_mask, :] = parents[1].brain.all_weights[i][~row_mask, :]
        new_weight2[~row_mask, :] = parents[0].brain.all_weights[i][~row_mask, :]
        new_weight2[row_mask, :] = parents[1].brain.all_weights[i][row_mask, :]

        all_weights1.append(new_weight1)
        all_weights2.append(new_weight2)

    child1 = dot_type()
    child2 = dot_type()

    child1.brain = SkipNeuralNet(all_weights=all_weights1, **parents[0].brain.get_configs())
    child2.brain = SkipNeuralNet(all_weights=all_weights1, **parents[0].brain.get_configs())

    return [child1, child2]


    
def dijkstra_on_output_crossover(parents : list[Dot], n_connections):


    parent1_neuron_pairs = parents[0].brain.neuron_pairs
    parent2_neuron_pairs = parents[1].brain.neuron_pairs

    n_connections_used = 0
    n_layers = len(parent1_neuron_pairs)

    child1_neuron_pairs = []

    layer_child = []
    layer = parent1_neuron_pairs[-1]
    for neuron in np.unique(layer):

        layer_child.append(layer[neuron == layer[:, 1]])

    NotImplemented

def gene_mix_crossover(parents : list[Dot], dot_type=DotGenetic) -> list[Dot]:
    
    
    genes_a = parents[0].genome.split('x')[:-1]
    debug_a = parents[0].genome.split('x')
    
    genes_b = parents[1].genome.split('x')[:-1]
    
    n_genes = len(genes_a)
    
    n_offspring = 2 
    new_offspring = []
    
    for ofspr in range(n_offspring):
        
        n_genes_from_a = np.random.randint(n_genes)
        n_genes_from_b = n_genes - n_genes_from_a
        
        genome = []
        genome += random.sample(genes_a, n_genes_from_a)
        genome += random.sample(genes_b, n_genes_from_b) 
        
        new_genome = 'x'.join(genome) + 'x'
        new_offspring.append(dot_type(id=1, genome=new_genome))
        
    return new_offspring           
    
def weights_mutation(dot_objects : list[Dot], flip_rate=0.1):

    if len(dot_objects) == 0:
        return []
    
    rng = np.random.default_rng()

    for dot in dot_objects:
        for i in range(len(dot.brain.all_weights)):
            
            # noise = rng.normal(size=dot.brain.all_weights[i].shape)
            noise_mask = (rng.random(size=dot.brain.all_weights[i].shape) < flip_rate)

            dot.brain.all_weights[i][noise_mask] += rng.normal(size=(np.count_nonzero(noise_mask),))
    
    return dot_objects

def bit_flip_mutation(new_dot_objects : list[Dot], flip_rate=0.1) -> list[Dot]:
    
    if len(new_dot_objects) == 0:
        return []
    rng = np.random.default_rng()
    
    len_genome = len(new_dot_objects[0].genome)
    
    mutant_dot_objects = new_dot_objects 
    
    
    for dot in mutant_dot_objects:
        old_genome = dot.genome
        new_genome = old_genome
        
        n_flips = rng.binomial(len_genome, flip_rate)
        for _ in range(n_flips):
            while True: # if random idx lands at 'x' --> loop doesnt break
                rnd_idx = rng.integers((len(old_genome)))
                
                if old_genome[rnd_idx] == '0':
                    new_genome = new_genome[:rnd_idx] + '1' + new_genome[rnd_idx + 1:] #string is immutable
                    break
                    
                elif old_genome[rnd_idx] == '1':
                    new_genome = new_genome[:rnd_idx] + '0' + new_genome[rnd_idx + 1:]
                    break
        
        dot.genome = new_genome
        
        
    return new_dot_objects

def create_offspring(parents : list[Dot], n_population, dot_type, species=None, crossover_func=one_point_crossover, mutation_func=bit_flip_mutation) -> list[Dot]:
    if len(parents) <= 1:
        raise Exception("not enough parents provided")
    
    crossed_dots = crossover(parents, crossover_func, n_population, dot_type=dot_type)
    mutated_dots = mutation_func(crossed_dots)
    offspring = mutated_dots
    
    for dot in offspring:
        dot.species = species 

    return offspring


def create_offspring_cm_seperat(parents : list[Dot], all_dots, n_population, dot_type, species=None, crossover_func=one_point_crossover, mutation_func=bit_flip_mutation, crossover_rate : float =0.8) -> list[Dot]:
    
    if len(parents) * crossover_rate <= 2:
        # n_crossover_parents = len(parents) if crossover_rate >= 0.5 else 0 
        parents = all_dots
        n_crossover_parents = 0
    else:
        n_crossover_parents = round(crossover_rate * len(parents))

    crossover_parents = parents[:n_crossover_parents]
    mutation_parents = parents[n_crossover_parents:]

    crossed_dots = []
    if n_crossover_parents >= 2:
        crossed_dots = crossover(crossover_parents, crossover_func, n_population - (len(parents) - n_crossover_parents), dot_type=dot_type)
    mutated_dots = mutation_func(mutation_parents)
    
    offspring = crossed_dots + mutated_dots
    for dot in offspring:
        dot.species = species 

    return offspring
