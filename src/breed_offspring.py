import numpy as np
import random 

from dot import Dot
from typing import Callable



    


def crossover(parent_objects : list[Dot], crossover_func : Callable, n_population : int, n_species=1, species_abs_size=None) -> list[Dot]:
    
    new_dot_objects = []
    
    n_parents = 2
    if len(parent_objects) < n_parents:
        raise Exception(f"only {len(parent_objects)} given, but at least expecting {n_parents}")

    if n_species > 1:
        
        parent_objects_species : list[list[Dot]] = []
        for species in range(1, n_species + 1):
            parent_objects_species.append([parent for parent in parent_objects if parent.species == species])
    
        for i, parent_objects in enumerate(parent_objects_species):
            
            n_childs = species_abs_size[i]
            
            parent_pairs : list[list[Dot]] = [random.sample(parent_objects, n_parents) for _ in range(int(n_childs / n_parents))]
            
            for pair in parent_pairs:

                offspring = crossover_func(pair)
                for o in offspring:
                    o.species = i + 1
                    
                new_dot_objects.append(offspring)
                
    else:
        for i in range(int(n_population / n_parents)):
        
            parents = random.sample(parent_objects, n_parents)
            
            offspring = crossover_func(parents)
            
            for o in offspring:
                new_dot_objects.append(o)
        
    return new_dot_objects
        
def one_point_crossover(parents : list[Dot]) -> list[Dot]:
    
    genome_len = len(parents[0].genome)

    idx = np.random.randint(genome_len) # x's between genes dont need to be stript
    
    head_a, tail_a = parents[0].genome[:idx], parents[0].genome[idx:]
    head_b, tail_b = parents[1].genome[:idx], parents[1].genome[idx:]
    
    offspring_a_genome = head_a + tail_b 
    offspring_b_genome = head_b + tail_a 
    
    new_offspring = []
    
    new_offspring.append(Dot(1, offspring_a_genome))
    new_offspring.append(Dot(1, offspring_b_genome))
    
    return new_offspring 

def gene_mix_crossover(parents : list[Dot]) -> list[Dot]:
    
    
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
        new_offspring.append(Dot(1, new_genome))
        
    return new_offspring           
    
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

def create_offspring(parents : list[Dot], all_dots, n_population, crossover_func=one_point_crossover, crossover_rate=0.8) -> list[Dot]:

    if len(parents) <= 1:
        parents = all_dots
        n_crossover_parents = 0
    elif len(parents) * crossover_rate <= 2:
        # n_crossover_parents = len(parents) if crossover_rate >= 0.5 else 0 
        parents = all_dots
        n_crossover_parents = 0
    else:
        n_crossover_parents = int(crossover_rate * len(parents)) + 1

    crossover_parents = parents[:n_crossover_parents]
    mutation_parents = parents[n_crossover_parents:]

    crossed_dots = []
    if n_crossover_parents >= 2:
        crossed_dots = crossover(crossover_parents, crossover_func, n_population - (len(parents) - n_crossover_parents))
    mutated_dots = bit_flip_mutation(mutation_parents)
    
    offspring = crossed_dots + mutated_dots
    for i, dot in enumerate(offspring):
        dot.id = i 

    return offspring
