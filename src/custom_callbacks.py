from callbacks_module import Callback

import os  
import numpy as np 
import math
from datetime import datetime
from time import perf_counter
import json 
import csv

class TimingCallback(Callback):
    pass
   
class TimeGens(TimingCallback):
    
    def __init__(self, print_time_on_gen_end : bool = False, printing_intervall : int = 5) -> None:
        super().__init__()
        
        self.print_time_on_gen_end : bool = print_time_on_gen_end
        self.time_first_gen_start : float | None = None
        self.gen_end_timestamps : list[float] = []
        self.printing_intervall : int = printing_intervall
        
    def on_run_begin(self, world):
            
        self.time_first_gen_start = perf_counter()
        self.gen_end_timestamps.append(self.time_first_gen_start)
            
            
    
    def on_gen_end(self, world):
        
        gen = world.current_gen

        t = perf_counter()
        
        self.gen_end_timestamps.append(t)

        if self.print_time_on_gen_end: 
            time_current_gen = t - self.gen_end_timestamps[-2]
            print(round(time_current_gen, 2), " s")    
        
        
            if gen % self.printing_intervall == 0:
                
                self.print_summary()
        

    def get_timing_summary(self):
        total_gens = len(self.gen_end_timestamps) - 1 # -1 because of first timestamp
        total_time = self.gen_end_timestamps[-1] - self.gen_end_timestamps[0]

        timing_dict = {"total gens": total_gens, 
                       "total time": total_time,  
                       "average secs per gen": total_time / total_gens} 
        return timing_dict
            
    
    def print_summary(self):
        
        data = self.get_timing_summary()
        total_time = data['total time']
        average_secs = data['average secs per gen']
        print(f"total time: {total_time}s\naverage time per gen: {average_secs}secs\n")

                     
class LoggingCallback(Callback):
    pass

    

class TimeGensLogger(TimeGens, LoggingCallback):
    def __init__(self, logging_intervall: int = 5) -> None:
        super().__init__(print_time_on_gen_end=False)
        self.logging_intervall = logging_intervall

    def log_time(self):
        data_dict = self.get_timing_summary()
        with open(f"performance/timing{data_dict['total gens']}.json", "w") as fp:
            json.dump(data_dict, fp, indent=4) 

    def on_run_begin(self, world):
        super().on_run_begin(world)

    def on_gen_end(self, world):
        super().on_gen_end(world)

        if world.current_gen % self.logging_intervall == 0:
            self.log_time()
    

class LoggingPointMixin(LoggingCallback):
    def __init__(self, log_points : str | list[int], n_max_gen : int, n_logs : int):
        
        if n_logs == 0:

            self.log_points = np.array([])
        
        elif log_points == 'linear':
            
            slope = int(n_max_gen / n_logs)
            
            self.log_points = np.round(slope * np.arange(1, n_logs + 1))
            
        elif log_points == 'polynomial':
            
            rank = math.log(n_max_gen, n_logs)
            
            self.log_points = np.round(np.arange(1, n_logs + 1) ** rank)
            
        elif log_points == 'exponential':
            
            base = n_max_gen ** (1 / n_logs) # nth-squareroot
            
            self.log_points = base ** np.arange(1, n_logs + 1) 
            
        elif isinstance(log_points, str):
            print(f'{log_points} not available')       
            
        else: # user can pass own list 
            
            self.log_points = log_points
            
        self.log_points = set(self.log_points)     

class LogWorldState(LoggingPointMixin):
    
    def __init__(self, log_points, n_max_gen, n_logs = 5, log_pos=False):
        
        super().__init__(log_points, n_max_gen, n_logs)
        
        self.log_pos = log_pos
        self.log_current_gen = False
    
    def on_run_begin(self, world):
        self.field_names = ["step", "total_in_zone"] + \
                        [f"in_zone_species_{i}" for i in range(1, world.n_species+1)]

        if world.kill_enabled:
            self.field_names += ['total_killed'] + [f"killed_species_{i}" for i in range(1, world.n_species+1)]
        

    def on_gen_begin(self, world):
        
        if world.current_gen in self.log_points:
            self.log_current_gen = True 
        else:
            self.log_current_gen = False 
        
        if self.log_current_gen: 
            gen = world.current_gen
            if not os.path.isdir(f"generations/gen{gen}"):
                os.mkdir(f"generations/gen{gen}")
                os.mkdir(f"generations/gen{gen}/statistics")
                with open(f"generations/gen{gen}/statistics/stats.csv", "a") as fp:
                    writer = csv.DictWriter(fp, fieldnames=self.field_names)
                    writer.writeheader()
                    writer.writerow({key:0 for key in self.field_names})
                    
            os.mkdir(f'generations/gen{gen}/step_pop_pos_species') 

            self.save_pop_pos(world)


                     
    def on_step_end(self, world):

        if self.log_current_gen:
            
            step = world.current_step
            gen = world.current_gen

            with open(f"generations/gen{gen}/statistics/stats.csv", "a") as fp:
                writer = csv.DictWriter(fp, fieldnames=self.field_names)
                alive_dots = [dot for dot in world.dot_objects if dot.alive]
                n_total_killed = world.n_population - len(alive_dots)
                n_total_in_zone = sum(world.death_func(world.pop_pos[[dot.id for dot in alive_dots], :])[0])
                n_killed_species = []
                n_in_zone_species = []
                for i in range(world.n_species):
                    alive_dots_species = [dot for dot in alive_dots if dot.species == i + 1]
                    n_killed_species.append(world.species_abs_size[i] - len(alive_dots_species))

                    pop_pos_species = world.pop_pos[[dot.id for dot in alive_dots_species], :]
                    is_in_zone, _ = world.death_func(pop_pos_species)
                    n_in_zone_species.append(sum(is_in_zone))
                
                # important: order needs to match order in self.field_names !
                row_vals = [step, n_total_in_zone] + n_in_zone_species
                if world.kill_enabled:
                    row_vals += [n_total_killed,] + n_killed_species 
                
                row = {key:val for key, val in zip(self.field_names, row_vals)}
                writer.writerow(row)
            
            self.save_pop_pos(world)
                
            # if self.log_pos:
            #     pop_pos = world.pop_pos
            #     np.savetxt(f'generations/gen{gen}/step_pop_pos/step{step}', pop_pos, fmt="%5i")
        
    def save_pop_pos(self, world):

        pop_pos = world.pop_pos
        species = np.array(world.species_list)
        
        pop_pos_species = np.c_[pop_pos, species] 

        np.savetxt(f'generations/gen{world.current_gen}/step_pop_pos_species/step{world.current_step}', pop_pos_species, fmt="%s")
                            
class WorldConverter:
    
    @staticmethod
    def species_to_tensor(run_dir : str):
        os.chdir(run_dir)
        world_configs = json.load('world_configurations/world_params.json')
        n_species : bool = world_configs['n_species']
        world_shape = world_configs['world_shape']
        world_state_zeros = np.zeros(world_shape, dtype=np.bool_)

        for i, gen in enumerate(os.listdir('generations')):
            
            os.mkdir(f'gen{i}/step_world_state_species')
            for j, step in enumerate(os.listdir(f'gen{i}/step_pop_pos_species')):

                world_state_list = []
                pop_pos_species = np.loadtxt(step)
                
                pop_pos_species_seperated = [(pop_pos_species == species)[:, :2] for species in range(1, n_species + 1)]
                for pos_species in pop_pos_species_seperated:
                    world_state = world_state_zeros.copy()[pos_species[:, 0], pos_species[:, 1]] = True 
                    world_state_list.append(world_state)

                world_state_3D = np.dstack(world_state_list)
                np.savetxt(f'gen{i}/step_world_state_species/step{j}', world_state_3D, fmt='%5i')
                                                 
class LogWeights(LoggingPointMixin): # world: flattened nn weights of all dots in in one 2d array
    
    def __init__(self, log_points, n_max_gen, n_logs = 5):
        
        super().__init__(log_points, n_max_gen, n_logs)
        
    def on_gen_begin(self, world):

        gen = world.current_gen
        
        if gen in self.log_points:
            dot_objects = world.dot_objects
            
            # init array 
            n_dif_inputs, n_dif_hidden, n_dif_outputs = world.n_dif_inputs, world.n_dif_hidden, world.n_dif_outputs 
            weight_sizes = np.array([0, n_dif_inputs * n_dif_hidden, n_dif_inputs * n_dif_outputs, 
                                     n_dif_hidden * n_dif_outputs])
            weight_sizes_cum = np.cumsum(weight_sizes)
            
            n_weights = np.sum(weight_sizes)
            n_dots = len(dot_objects)
            
            weights_array = np.empty((n_dots, n_weights))
            
            for i, dot in enumerate(dot_objects):
                
                for w_i, weight in enumerate(dot.brain.all_weights): 
                    
                    j = weight_sizes_cum[w_i]
                    try:   #TODO make less ugly
                        j_next = weight_sizes_cum[w_i + 1]
                    except IndexError:
                        weights_array[i, j:] = weight.flatten()
                    else:  
                        weights_array[i, j:j_next] = weight.flatten()
                        
            if not os.path.isdir(f"generations/gen{gen}"):
                os.mkdir(f'generations/gen{gen}')
            os.mkdir(f'generations/gen{gen}/weights')
            np.savetxt(f'generations/gen{gen}/weights/all_weights', weights_array)

class InitLogs(LoggingCallback):

    def on_init_simulation(self, world):
        time = str(datetime.now()).split(".")[0].replace(':', ' ')

        current_wdir = os.getcwd()

        dir_path = os.path.join(current_wdir, 'logs', f'run_{time}')

        os.mkdir(dir_path)
            
        os.chdir(dir_path) 

        os.mkdir("world_configurations")
        os.mkdir("generations")
        os.mkdir("colorizing_data")
        os.mkdir("performance")

        np.savetxt("world_configurations/death_mask", world.death_func.mask, fmt="%5i")
        if world.obstacles_enabled:
            np.savetxt("world_configurations/wall_mask", world.wall_mask, fmt="%5i")

        with open("world_configurations/world_params.json", "w") as fp:
            param_dict = world.get_config()
            json.dump(param_dict, fp, indent=4) 
                    
                    