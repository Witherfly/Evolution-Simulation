import numpy as np
import random 
import os
import json 
import datetime 
import multiprocessing as mp

from dot import Dot, DotGenetic
from breed_offspring import create_offspring, one_point_crossover, bit_flip_mutation
from live_plot import plot, PlottingStatCollector, StatCollector
from live_plot_mp import PlotterMP
from callbacks_module import Callback, CallbackList
from custom_callbacks import LoggingCallback


from collections.abc import Callable
import numpy.typing as npt 
from typing import Any 

from line_profiler import profile



    
 
class World():
    
    def __init__(self, world_shape, n_population, n_steps, n_max_gen, 
                 n_connections, create_logs,
                 death_func : Callable[[np.ndarray], tuple[np.ndarray, dict[str , int] | None]],
                 dot_type=DotGenetic,
                 crossover_func=one_point_crossover,
                 mutation_func=bit_flip_mutation,
                 n_neurons_hidden_layer : tuple[int, ...] = (8,),
                 live_plotting=True,
                 plotting_stat_collector : PlottingStatCollector | None = None,
                 no_spawn_in_zone=False, 
                 kill_enabled=False, 
                 wall_mask : npt.NDArray[np.bool_] | None = None, 
                 random_state=42, 
                 callbacks : list[Callback] | None = None,
                 n_species : int = 1,
                 trans_species_killing : str = 'no_restriction', # domestic_only, foreign_only 
                 species_obs : bool = False,
                 dot_density_obs : bool = True,
                 species_rel_size : tuple[float, ...] | None = None,
                 
                 ):
        
        self.world_shape = world_shape
        self.n_population = n_population
        self.current_gen = 0
        self.current_step = 0 
        self.n_steps = n_steps
        self.dot_type = dot_type
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.n_max_gen = n_max_gen
        self.create_logs = create_logs
        self.death_func = death_func
        self.no_spawn_in_zone = no_spawn_in_zone
        self.kill_enabled = kill_enabled
        self.wall_mask = wall_mask 
        self.n_species = n_species
        self.live_plotting = live_plotting
        self.live_plotting_mp = True
        self.plotting_stat_collector = plotting_stat_collector 
        if plotting_stat_collector is None:
            self.plotting_stat_collector = StatCollector()
        if trans_species_killing not in ['domestic_only', 'no_restriction', 'foreign_only']:
            raise Exception(trans_species_killing)
        self.trans_species_killing = trans_species_killing if n_species > 1 else 'no_restriction'
        self.species_obs = species_obs if n_species > 1 else False
        self.dot_density_obs = dot_density_obs
        self.dot_density_mask_left = np.array([[0, 0, 1, 1],
                                               [0, 1, 1, 1],
                                               [1, 1, 1, 0],
                                               [0, 1, 1, 1],
                                               [0, 0, 1, 1]], dtype=np.bool_)
        self.dot_density_mask_down = np.rot90(self.dot_density_mask_left, 1)
        self.dot_density_mask_right = np.rot90(self.dot_density_mask_left, 2)
        self.dot_density_mask_up = np.rot90(self.dot_density_mask_left, 3)
        self.dot_density_mask_size = np.count_nonzero(self.dot_density_mask_left)

        self.world_padding : int = max(self.dot_density_mask_left.shape[0], self.dot_density_mask_left.shape[1]//2)

        if species_rel_size is None: # all species have the same size 
            self.species_rel_size = tuple([1 / self.n_species,] * self.n_species)
        else:
            self.species_rel_size = species_rel_size

        self.species_abs_size = [int(i*self.n_population) for i in self.species_rel_size]
        for i in range(self.n_population - sum(self.species_abs_size)): 
            self.species_abs_size[i] += 1
        
        self.species_abs_size = tuple(self.species_abs_size)
        
        # callbacks 
        self.callbacks = CallbackList(callbacks)
        if not self.create_logs:
            self.callbacks.remove_callbacks_type(LoggingCallback)
        if callbacks is None:
            self.callbacks = CallbackList()
        
        
        self.rnd_seed = random_state
        
        self.obstacles_enabled = True
        if self.wall_mask is None or np.all(self.wall_mask == 0):
            self.obstacles_enabled = False
        
        self.world_size = self.world_shape[0] * self.world_shape[1]
        
        
        self.pop_pos = np.empty((self.n_population, 2), dtype=np.int8)
        self.world_state = np.zeros(self.world_shape, dtype=np.int8)
        
        self.n_connections = n_connections
        self.n_dif_inputs = 9
        self.n_dif_outputs = 4
        
        if self.kill_enabled:
            self.n_dif_outputs += 4
        if self.obstacles_enabled:
            self.n_dif_inputs += 4
        if self.species_obs:
            self.n_dif_inputs += 4 
        if self.dot_density_obs:
            self.n_dif_inputs += 4
        
        self.n_neurons_per_layer = (self.n_dif_inputs, *n_neurons_hidden_layer, self.n_dif_outputs)
        # plotting
        self.n_killed_list : list[int] = []
        self.n_survived_list : list[int] = []
        
        
        
        self.plot_dict : dict[str , list[float]] = {}

    @classmethod
    def init_from_json(cls, json_file):  # TODO extract python object from json file not implemented
        
        json_dict = json.load(json_file)
        
        return cls(**json_dict)
    
    def get_config(self) -> dict[str, Any]:
            
        log_var_list = ["world_shape", "n_population", 'n_steps', 'n_max_gen', 
            'n_connections', 'create_logs', 'n_neurons_per_layer',
            'kill_enabled', 'no_spawn_in_zone', 'n_species', 'trans_species_killing']
        
        param_dict = {key:self.__dict__[key] for key in log_var_list}
        ## param_dict = self.__dict__ 
                
        return param_dict 
            
    def place_pop(self):
        
        #np.seed(self.rnd_seed)
        
        world_idx_flatt = np.arange(self.world_size)
        
        if self.obstacles_enabled:
            wall_idx = np.ravel(np.argwhere(np.ravel(self.wall_mask)))
        else:
            wall_idx = np.array([], dtype=np.int32)
            
            
        if self.no_spawn_in_zone:
            zone_idx = np.ravel(np.argwhere(np.ravel(self.death_func.mask)))
        else:
            zone_idx = np.array([], dtype=np.int32)
            
        if self.obstacles_enabled or self.no_spawn_in_zone:       
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
        
        species_thresh = np.cumsum(self.species_abs_size)
        
        for i in range(self.n_population):

            species = np.sum(species_thresh <= i) + 1
            dot = self.dot_type(id=i, species=species)
            dot.random_init(self.n_connections, self.n_neurons_per_layer)
            
            self.dot_objects.append(dot)   
            

        if self.live_plotting and self.live_plotting_mp:
            self.plot_live_mp_init()
        
        self.callbacks.on_init_simulation(self)
            

    @staticmethod
    def selection(dot_objects : list[Dot], survived : npt.NDArray[np.bool_], stat_collector=None):

        parent_objects = []
        for i, dot in enumerate(dot_objects):
            stat_collector(dot, survived[i])
            if not dot.alive:
                continue
            if survived[i]:
                parent_objects.append(dot)

        return parent_objects
    
    def make_next_generation(self, parent_objects) -> list[Dot]:

        new_dot_objects = [] 
        for i in range(1, self.n_species + 1):
            parent_objects_species = [d for d in parent_objects if d.species == i]
            if len(parent_objects_species) <= 1:
                parent_objects_species = [d for d in self.dot_objects if d.species == i]
            offspring = create_offspring(parent_objects_species, 
                                            n_population=self.species_abs_size[i-1], 
                                            species=i, 
                                            dot_type=self.dot_type,
                                            crossover_func=self.crossover_func,
                                            mutation_func=self.mutation_func)
            new_dot_objects += offspring

        assert len(new_dot_objects) == self.n_population

        for i, dot in enumerate(new_dot_objects):
            dot.unencode_genome(self.n_neurons_per_layer)
            dot.id = i
            dot.alive = True
        
        return new_dot_objects

    
    def start_simulation(self):
        
        self.callbacks.on_run_begin(self) 
        
        for gen in range(1, self.n_max_gen + 1):
            
            print(gen)
            self.current_step = 0
            self.current_gen = gen
            self.place_pop()
            
            self.callbacks.on_gen_begin(self)

            for step in range(1, self.n_steps + 1):
            
                
                self.current_step = step
                self.world_state_padded = np.pad(self.world_state, self.world_padding)

                for i, dot in enumerate(self.dot_objects): 
                    
                    if dot.alive:
                        inputs = self.create_observation(dot.id)
                        action = dot.move(inputs)
                        
                        self.apply_action(dot.id, action) # grid is updated here individually

                
                self.callbacks.on_step_end(self)
     
            is_alive, zone_info_dict = self.death_func(self.pop_pos) 
            
            parent_objects = self.selection(self.dot_objects, is_alive, self.plotting_stat_collector)
                    
            self.dot_objects = self.make_next_generation(parent_objects)
            
            if self.live_plotting:

                if self.live_plotting_mp:
                    self.plot_live_mp(zone_info_dict)
                else:
                    self.plot_live(zone_info_dict)
            
            self.callbacks.on_gen_end(self)
    
    def plot_live(self, zone_info_dict):

        if zone_info_dict is not None:
            for key in zone_info_dict.keys():
                if self.current_gen==1:
                    if key not in self.plot_dict.keys():
                        self.plot_dict.update({key: []})
                    
            
                self.plot_dict[key].append(zone_info_dict[key] / self.n_population)
        
        else:
            self.plot_dict = None 

        self.n_survived_list.append(self.plotting_stat_collector.survived_counts)
        rel_survivors = np.array(self.n_survived_list) / self.species_abs_size

        rel_killed = None 
        if self.kill_enabled:
            self.n_killed_list.append(self.plotting_stat_collector.killed_counts)
            rel_killed = np.array(self.n_killed_list) / self.species_abs_size

        self.plotting_stat_collector.reset()
        
        is_last_gen = False if self.current_gen != self.n_max_gen else True 

        plot(is_last_gen, rel_survivors, self.plot_dict, killed=rel_killed)
    
    def plot_live_mp_init(self):

        self.plot_queue_mp = mp.Queue()
        self.plotter_mp = PlotterMP()
        self.plot_process = mp.Process(target=self.plotter_mp.start, args=(self.plot_queue_mp,), daemon=True)
        self.plot_process.start()

    def plot_live_mp(self, zone_info_dict):
        
        data = {'survived' : self.plotting_stat_collector.survived_counts / self.species_abs_size}
        if self.kill_enabled:
            data['killed'] = self.plotting_stat_collector.killed_counts / self.species_abs_size
        
        self.plot_queue_mp.put(data)

        self.plotting_stat_collector.reset()
        
        is_last_gen = False if self.current_gen != self.n_max_gen else True 
        if is_last_gen:
            self.plot_queue_mp.put(None)
            self.plot_process.join()
            self.plot_process.kill()
     
    def create_observation(self, id : int) -> npt.NDArray[np.float32]:
        
        
        dot = self.dot_objects[id]
        x, y = self.pop_pos[id] # note: at top x is 0, at bottom its positiv, so x and y are swapped
        step_frac = self.current_step / self.n_steps
        oscillator1 = self.current_step % 2
        oscillator2 = self.current_step % 3
        obs_list = [step_frac, oscillator1, oscillator2]
        
        north_distance = x 
        west_distance = y 
        # division for normalization between 0 and 1
        north_distance_norm = north_distance / (self.world_shape[0] - 1)
        west_distance_norm =  west_distance / (self.world_shape[1] - 1)
        
        nw_distances = [north_distance_norm, west_distance_norm]
        
        obs_list += nw_distances
        
        try:
            north_blocked = self.world_state[x - 1, y] if x - 1 >= 0 else 0
        except IndexError:
            north_blocked = 0
        try:
            south_blocked = self.world_state[x + 1, y] 
        except IndexError:
            south_blocked = 0
        try:
            west_blocked = self.world_state[x , y - 1] if y - 1 >= 0 else 0
        except IndexError:
            west_blocked = 0
        try:
            east_blocked = self.world_state[x , y + 1]
        except IndexError:
            east_blocked = 0
        
        nswe_blocked = [north_blocked, south_blocked, west_blocked, east_blocked]
        
        obs_list += nswe_blocked

        if self.dot_density_obs:
            x, y = x + self.world_padding, y + self.world_padding
            dot_density_left = np.sum(self.world_state_padded[x-2:x+3, y-4:y][self.dot_density_mask_left]) / self.dot_density_mask_size
            dot_density_down = np.sum(self.world_state_padded[x+1:x+5, y-2:y+3][self.dot_density_mask_down]) / self.dot_density_mask_size
            dot_density_right = np.sum(self.world_state_padded[x-2:x+3, y+1:y+5][self.dot_density_mask_right]) / self.dot_density_mask_size
            dot_density_up = np.sum(self.world_state_padded[x-4:x, y-2:y+3][self.dot_density_mask_up]) / self.dot_density_mask_size
            
            obs_list += [dot_density_left, dot_density_down, dot_density_right, dot_density_up]
            x, y = x - self.world_padding, y - self.world_padding
        
        
        if self.obstacles_enabled:
            try:
                north_wall = self.wall_mask[x - 1, y]
            except IndexError:
                north_wall = 1
            try:
                south_wall = self.wall_mask[x + 1, y]
            except IndexError:
                south_wall = 1
            try:
                west_wall = self.wall_mask[x , y - 1]
            except IndexError:
                west_wall = 1
            try:
                east_wall = self.wall_mask[x , y + 1]
            except IndexError:
                east_wall = 1
            
            nswe_wall = [north_wall, south_wall, west_wall, east_wall]
            obs_list += nswe_wall
            
        if self.species_obs:

            north_species, south_species, east_species, west_species = False, False, False, False
            
            if north_blocked:            
                north_species = self.dot_at_pos((x-1, y)).species == dot.species
            if south_blocked:
                south_species = self.dot_at_pos((x+1, y)).species == dot.species
            if west_blocked:
                west_species = self.dot_at_pos((x, y-1)).species == dot.species
            if east_blocked:
                east_species = self.dot_at_pos((x, y+1)).species == dot.species
            
            species_obs_array = [north_species, south_species, west_species, east_species]
            
            obs_list += species_obs_array

        obs_array = np.array(obs_list) 
        
        return obs_array
    
    def dot_at_pos(self, pos: tuple[int, int], check_occ=False) -> Dot | None:
        
        if check_occ and pos[0] < 0 or pos[1] < 0 or pos[0] >= self.world_shape[0] or pos[1] >= self.world_shape[1]:
            return None
        if check_occ and self.world_state[pos[0], pos[1]] == 0:
            return None
        # idx = np.where(np.sum(self.pop_pos == pos, axis=1) == 2)[0].item()
        idx = np.argwhere(np.logical_and(self.pop_pos[:, 0] == pos[0], self.pop_pos[:, 1] == pos[1])).item()
        return self.dot_objects[idx]
        
    def apply_action(self, id : int, action):
        
        
        current_pos = self.pop_pos[id]
        new_pos = np.copy(current_pos)
        kill_pos : tuple[int, int] | None = None 
         
        match action:
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

            
            
        if self.is_square_free(new_pos):
            #updating pos 
            self.world_state[current_pos[0], current_pos[1]] = 0
            self.world_state[new_pos[0], new_pos[1]] = 1
            
            self.pop_pos[id] = new_pos 

    def is_square_free(self, coords) -> bool:
        if coords[1] >= self.world_shape[1] or coords[0] >= self.world_shape[0] or coords[0] < 0 or coords[1] < 0: #for performance in one single line
            return False
            
        elif self.world_state[coords[0], coords[1]] == 1: #square already ocupied by other dot
            return False
        
        elif self.obstacles_enabled: # square occupied by wall tile 
            if self.wall_mask[coords[0], coords[1]] == 1:
                return False
        
        return True 
                
    def kill(self, kill_pos : tuple[int, int], killer_id : int) -> bool: # returns False if no kill happened
        
        victim = self.dot_at_pos((kill_pos[0], kill_pos[1]), check_occ=True)

        if victim is None: #no dot found at kill pos
            return False 

        killer = self.dot_objects[killer_id]
        
        match self.trans_species_killing:
            case 'no_restriction': # No restriction 
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
    
    @property
    def species_list(self):
        
        return [dot.species for dot in self.dot_objects]
    