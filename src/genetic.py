import numpy as np
import random 
import os
import json 
import datetime 

from .dot import Dot
from .live_plot import plot 
from . import callbacks_module


from collections.abc import Callable
import numpy.typing as npt 
from typing import Any 

        
 
class World():
    
    def __init__(self, world_shape, n_population, n_steps, n_max_gen, 
                 mutation_rate, flip_rate, n_connections, create_logs,
                 death_func : Callable[[np.ndarray], tuple[np.ndarray, dict[str , int] | None]]
                 ,no_spawn_in_zone=False, kill_enabled=False, 
                 wall_mask : npt.NDArray[np.bool_] | None = None, 
                 crossover_func_name="one_point_crossover",
                 random_state=42, callbacks : list[callbacks_module.Callback] | None = None,
                 n_species : int = 1,
                 trans_species_killing : str = 'no_restriction', # domestic_only, foreign_only 
                 species_obs : bool = False,
                 species_rel_size : tuple[float, ...] | None = None,
                 
                 ):
        
        self.world_shape = world_shape
        self.n_population = n_population
        self.current_gen = 0
        self.current_step = 0 
        self.n_steps = n_steps
        self.n_max_gen = n_max_gen
        self.mutation_rate = mutation_rate
        self.flip_rate = flip_rate 
        self.create_logs = create_logs
        self.death_func = death_func
        self.no_spawn_in_zone = no_spawn_in_zone
        self.kill_enabled = kill_enabled
        self.wall_mask = wall_mask 
        self.n_species = n_species
        if self.n_species == 1:
            if species_obs or species_rel_size is not None or trans_species_killing != 'no_restriction':
                raise Exception("multi_species is off but species-specificts were passed as arguments")
        self.trans_species_killing = trans_species_killing
        self.species_obs = species_obs
        if species_rel_size is None: # all species have the same size 
            species_size = int(self.n_population / n_species)
            species_size_mod = self.n_population % n_species
            self.species_rel_size = [species_size,] * self.n_species
                
            if species_size_mod > 0:
                for i in range(species_size_mod):
                    self.species_rel_size[i] += 1

        else:
            self.species_rel_size = species_rel_size
            
        self.species_rel_size = tuple(self.species_rel_size)
        self.species_abs_size = np.array(self.species_rel_size) * self.n_population
            
        
        # callbacks 
        self.callbacks = callbacks_module.CallbackList(callbacks)
        if not self.create_logs:
            self.callbacks.remove_logging_callbacks()
        if callbacks is None:
            self.callbacks = callbacks_module.CallbackList()
        
        crossover_func_dict = {"one_point_crossover":self.one_point_crossover,
                               "gene_mix_crossover": self.gene_mix_crossover}
        
        self.crossover_func = crossover_func_dict[crossover_func_name]
        
        self.rnd_seed = random_state
        
        self.obstacles_enababled = False if self.wall_mask is None or False else True 
        
        self.world_size = self.world_shape[0] * self.world_shape[1]
        
        
        self.pop_pos = np.empty((self.n_population, 2), dtype=np.int8)
        self.world_state = np.zeros(self.world_shape, dtype=np.int8)
        
        self.n_connections = n_connections
        self.n_dif_inputs = 6
        self.n_dif_hidden = 3
        self.n_dif_outputs = 4
        
        if self.kill_enabled:
            self.n_dif_outputs += 4
        if self.obstacles_enababled:
            self.n_dif_inputs += 4
        if self.species_obs:
            self.n_dif_inputs += 4 
        
        # plotting
        
        self.n_survivors_list = []
        self.n_killed_list = []
        self.n_killed = 0
        
        self.plot_dict : dict[str , list[float]] = {}

    @classmethod
    def init_from_json(cls, json_file):  # TODO extract python object from json file not implemented
        
        json_dict = json.load(json_file)
        
        return cls(**json_dict)
    
    def get_config(self) -> dict[str, Any]:
            
        log_var_list = ["world_shape", "n_population", 'n_steps', 'n_max_gen', 
            'mutation_rate', 'flip_rate', 'n_connections', 'create_logs', 
            'kill_enabled', 'no_spawn_in_zone', 'n_species', 'trans_species_killing']
        
        param_dict = {key:self.__dict__[key] for key in log_var_list}
        ## param_dict = self.__dict__ 
                
        return param_dict 
            
    def place_pop(self):
        
        #np.seed(self.rnd_seed)
        
        world_idx_flatt = np.arange(self.world_size)
        
        if self.obstacles_enababled:
            wall_idx = np.ravel(np.argwhere(np.ravel(self.wall_mask)))
        else:
            wall_idx = np.array([], dtype=np.int32)
            
            
        if self.no_spawn_in_zone:
            zone_idx = np.ravel(np.argwhere(np.ravel(self.death_func.mask)))
        else:
            zone_idx = np.array([], dtype=np.int32)
            
        if self.obstacles_enababled or self.no_spawn_in_zone:       
            world_idx_flatt = np.delete(world_idx_flatt, np.r_[wall_idx, zone_idx])
            

        world_idx_flatt_shuffled = np.random.permutation(world_idx_flatt)
        
        idxs_flatt = world_idx_flatt_shuffled[:self.n_population]
        
        
        idx_2D_0 = np.floor(idxs_flatt / self.world_shape[1]).astype(np.int8)
        idx_2D_1 = idxs_flatt % self.world_shape[1]
        
        self.pop_pos[:, 0] = idx_2D_0
        self.pop_pos[:, 1] = idx_2D_1 
        
        self.world_state *= 0
        self.world_state[idx_2D_0, idx_2D_1] = 1
        
    def init_simulation(self):
        
        self.place_pop()
        
        self.dot_objects : list[Dot] = []
        
        if self.n_species > 1:
            species_thresh = np.cumsum(np.array(self.species_rel_size) * self.n_population)
        
        species = None 
        for i in range(self.n_population):

            if self.n_species > 1:
                species = np.sum(species_thresh > i) + 1
            genome = Dot.create_genome(self.n_connections, self.n_dif_inputs, self.n_dif_hidden, self.n_dif_outputs)
            dot = Dot(i , genome, species)
            dot.unencode_genome(self.n_dif_inputs, self.n_dif_hidden, self.n_dif_outputs)
            
            self.dot_objects.append(dot)   
            
        # directorys for storing data
        
        self.callbacks.on_init_simulation(self)
            

                     
    def selection(self, is_alive : npt.NDArray[np.bool_]) -> list[Dot]:
        
        parent_objects  = [dot for i, dot in enumerate(self.dot_objects) if is_alive[i] and dot.alive]
    
        return parent_objects
    
    def crossover(self, parent_objects : list[Dot]) -> list[Dot]:
        
        new_dot_objects = []
        
        n_survivors = len(parent_objects)   # case of survivors == 1 or 0      
        self.n_survivors_list.append(n_survivors)
        
        n_parents = 2

        if self.n_species > 1:
            
            parent_objects_species : list[list[Dot]] = []
            for species in range(1, self.n_species + 1):
                parent_objects_species.append([parent for parent in parent_objects if parent.species == species])
        
            for i, parent_objects in enumerate(parent_objects_species):
                
                n_childs = self.species_abs_size[i]
                if len(parent_objects) < n_parents:
                    parent_objects = [dot for dot in self.dot_objects if dot.species == i + 1]
                
                parent_pairs : list[list[Dot]] = [random.sample(parent_objects, n_parents) for _ in range(int(n_childs / n_parents))]
                
                for pair in parent_pairs:

                    offspring = self.crossover_func(pair)
                    for o in offspring:
                        o.species = i + 1
                        
                    new_dot_objects.append(offspring)
                    
        else:
            for i in range(int(self.n_population / n_parents)):
            
                if len(parent_objects) > 1:
                    parents = random.sample(parent_objects, n_parents)
                elif len(parent_objects) == 1:
                    parents = parent_objects 
                    parents.append(random.choice(self.dot_objects))
                
                else:
                    print("no survivors")
                    parents = random.sample(self.dot_objects, n_parents)
                    
                
                offspring = self.crossover_func(parents)
                
                for o in offspring:
                    new_dot_objects.append(o)
            
                
         
               
        # assign correct id to objects       
        for i, dot in enumerate(new_dot_objects):
            
            dot.id = i 
            
        return new_dot_objects
          
    def one_point_crossover(self, parents : list[Dot]) -> list[Dot]:
        
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
    
    def gene_mix_crossover(self, parents : list[Dot]) -> list[Dot]:
        
        
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
        
    def bit_flip_mutation(self, new_dot_objects : list[Dot]) -> list[Dot]:
        
        
        len_genome = len(new_dot_objects[0].genome)
        # approximate binomial distribution with gaussian
        mean = self.n_population*self.mutation_rate
        standard_deviation = np.sqrt(mean * (1 - self.mutation_rate))
        n_mutants = int(np.round(np.random.normal(loc=mean, scale=standard_deviation)))
        
        n_mutants = n_mutants if n_mutants >= 0 else 0
       
        mutant_dot_objects = random.sample(new_dot_objects, n_mutants)
        
        
        
        mean_flips = len_genome * self.flip_rate 
        standard_deviation_flips = np.sqrt(mean * (1 - mean_flips / len_genome) )
        
        
        
        for dot in mutant_dot_objects:
            old_genome = dot.genome
            new_genome = old_genome
            
            n_flips = int(np.round(np.random.normal(loc=mean_flips, scale=standard_deviation_flips)))
            for _ in range(n_flips):
                while True: # if random idx lands at 'x' --> loop doesnt break
                    rnd_idx = np.random.randint((len(old_genome)))
                    
                    if old_genome[rnd_idx] == '0':
                        new_genome = new_genome[:rnd_idx] + '1' + new_genome[rnd_idx + 1:] #string is immutable
                        break
                        
                    elif old_genome[rnd_idx] == '1':
                        new_genome = new_genome[:rnd_idx] + '0' + new_genome[rnd_idx + 1:]
                        break
            
            dot.genome = new_genome
            
            
        return new_dot_objects

    def start_simulation(self):
        
        self.callbacks.on_run_begin(self) 
        
        for gen in range(1, self.n_max_gen + 1):
            
            print(gen)
            self.current_gen = gen
            self.callbacks.on_gen_begin(self)

            
            
            
            self.place_pop()
            for step in range(1, self.n_steps + 1):
                
                self.current_step = step
                for dot in self.dot_objects: # make more fair by shuffling object list 
                    
                    if dot.alive:
                        inputs = self.create_observation(dot.id)
                        
                        action = dot.move(inputs)
                        
                        self.apply_action(dot.id, action) # grid is updated here individually

                # if self.create_logs:   
                #     #log(gen, step)
                #     pass 
                
                self.callbacks.on_step_end(self)
                    
            is_alive, zone_info_dict = self.death_func(self.pop_pos) 
            
            parent_objects = self.selection(is_alive)
            
            new_dot_objects = self.crossover(parent_objects) 
            
            self.dot_objects = self.bit_flip_mutation(new_dot_objects)
            
            for dot in self.dot_objects:
                if dot.alive:
                    dot.unencode_genome(self.n_dif_inputs, self.n_dif_hidden, self.n_dif_outputs)
                    
            # plotting
            
            self.n_killed_list.append(self.n_killed)
            self.n_killed = 0 
            
            if zone_info_dict is not None:
                for key in zone_info_dict.keys():
                    if gen==1:
                        if key not in self.plot_dict.keys():
                            self.plot_dict.update({key: []})
                        
                
                    self.plot_dict[key].append(zone_info_dict[key] / self.n_population)
            
            else:
                self.plot_dict = None 
                
                
            
            
            
            
            rel_survivors = np.array(self.n_survivors_list) / self.n_population
            rel_killed = np.array(self.n_killed_list) / self.n_population
            
            is_last_gen = False if gen != self.n_max_gen else True 
            
            plot(is_last_gen, rel_survivors, rel_killed, self.plot_dict)

            
            self.callbacks.on_gen_end(self)
     
    def create_observation(self, id : int) -> npt.NDArray[np.float32]:
        
        
        dot = self.dot_objects[id]
        x, y = self.pop_pos[id] # note: at top x is 0, at bottom its positiv
        obs_list = []
        
        north_distance = x # division for normalization between 0 and 1
        west_distance = y 
        
        north_distance_norm = north_distance / self.world_shape[0]
        west_distance_norm = west_distance / self.world_shape[1]
        
        nw_distances = [north_distance_norm, west_distance_norm]
        
        obs_list += nw_distances
        
        try:
            north_blocked = self.world_state[x - 1, y]
            south_blocked = self.world_state[x + 1, y]
        except IndexError:
            north_blocked, south_blocked = 0, 0
            
            
        try:
            east_blocked = self.world_state[x , y + 1]
            west_blocked = self.world_state[x , y - 1]
        except IndexError:
            east_blocked, west_blocked = 0, 0
        
        nsew_blocked = [north_blocked, south_blocked, east_blocked, west_blocked]
        
        obs_list += nsew_blocked
        
        
        
        if self.obstacles_enababled:
            try:
                north_wall = self.wall_mask[x - 1, y]
                south_wall = self.wall_mask[x + 1, y]
            except IndexError:
                north_wall, south_wall = 1, 1
                
                
            try:
                east_wall = self.wall_mask[x , y + 1]
                west_wall = self.wall_mask[x , y - 1]
            except IndexError:
                east_wall, west_wall = 1, 1 
            
            nsew_wall = [north_wall, south_wall, east_wall, west_wall]
            obs_list += nsew_wall
            
               
                

        if self.species_obs:

            north_species, south_species, east_species, west_species = False, False, False, False
            
            if north_blocked:            
                north_species = self.dot_at_pos((x, y - 1)).species == dot.species
            if south_blocked:
                south_species = self.dot_at_pos((x, y + 1)).species == dot.species
            if east_blocked:
                east_species = self.dot_at_pos((x + 1, y)).species == dot.species
            if west_blocked:
                west_species = self.dot_at_pos((x - 1, y)).species == dot.species
            
            species_obs_array = [north_species, south_species, east_species, west_species]
            
            obs_list += species_obs_array

       

        obs_array = np.array(obs_list) 
        
        return obs_array
    
    def dot_at_pos(self, pos: tuple[int, int], check_occ=False) -> Dot | None:
        
        if check_occ and pos[0] < 0 or pos[1] < 0 or pos[0] >= self.world_shape[0] or pos[1] >= self.world_shape[1]:
            return None
        if check_occ and self.world_state[pos[0], pos[1]] == 0:
            return None
        id : int = np.where(np.sum(self.pop_pos == pos, axis=1) == 2)[0].item()
        return self.dot_objects[id]
        
    @property
    def species_list(self):
        
        return [dot.species for dot in self.dot_objects]
    
    
    
    
    def apply_action(self, id : int, action : npt.NDArray[np.bool_]):
        
        
        current_pos = self.pop_pos[id]
        new_pos = np.copy(current_pos)
        kill_pos : tuple[int, int] | None = None 
         
        action_number : int = np.argwhere(action).item()
        match action_number:
            case 0:
                new_pos[0] = current_pos[0] - 1 
            case 1:
                new_pos[1] = current_pos[1] + 1
            case 2:
                new_pos[0] = current_pos[0] + 1
            case 3:
                new_pos[1] = current_pos[1] - 1
            #killing
            case 4:
                kill_pos = (current_pos[0] - 1, current_pos[1]) #north
            case 5:
                kill_pos = (current_pos[0] , current_pos[1] + 1) #east 
            case 6:
                kill_pos = (current_pos[0] + 1, current_pos[1]) #south
            case 7:
                kill_pos = (current_pos[0] , current_pos[1] - 1) #west
                
        if kill_pos is not None and self.kill_enabled: 
            was_killed = self.kill(kill_pos, killer_id=id)
            if was_killed: self.n_killed += 1 
                    
        def is_square_free(coords) -> bool:
            if coords[1] >= self.world_shape[1] or coords[0] >= self.world_shape[0]: # out of bound bottom and left
                return False
            elif np.any(coords < 0): # out of bound top and right 
                return False
            elif self.world_state[coords[0], coords[1]] == 1: #square already ocupied by other dot
                return False
            
            elif self.obstacles_enababled: # square occupied by wall tile 
                if self.wall_mask[coords[0], coords[1]] == 1:
                    return False
            
            return True 
            
            
        if is_square_free(new_pos):
            #updating pos 
            self.world_state[current_pos[0], current_pos[1]] = 0
            self.world_state[new_pos[0], new_pos[1]] = 1
            
            self.pop_pos[id] = new_pos 
                
           
        
            
           
    def kill(self, kill_pos : tuple[int, int], killer_id : int) -> bool: # returns False if no kill happened
        
        victim = self.dot_at_pos((kill_pos[0], kill_pos[1]), check_occ=True)

        if victim is None: #no dot found at kill pos
            return False 

        killer = self.dot_objects[killer_id]
        
        if self.n_species > 1:
            match self.trans_species_killing:
                case 'no_restricion': # No restriction 
                    pass 
                case 'domestic_only':
                    if victim.species != killer.species: # no kill if not same species 
                        return False 
                case 'foreign_only':
                    if victim.species == killer.species: # no kill if same species 
                        return False 
         
        victim.alive = False 
        self.world_state[kill_pos[0], kill_pos[1]] = 0
        self.pop_pos[victim.id] = np.array([-2, -2]) 
        
        return True 
    
