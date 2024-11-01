import os  
import numpy as np 
from typing import Union 
import math 
from time import perf_counter
from datetime import timedelta 
import json 

from typing import Any

class Callback:
    
    callback_points = ['on_simulation_init', 'on_step_end', 'on_gen_end', 'on_gen_begin', 'on_run_begin', 'on_init_simulation']

    def __repr__(self):
        cls = self.__class__
        cls_name = cls.__name__
        indent = ' ' * 4
        res = [f'{cls_name}('] 
        for name, value in self.__dict__.items():
            res.append(f'{indent}{name} = {value!r},') 
            
        res.append(')') 
        return '\n'.join(res) 
    
class CallbackList():
    
    """Container for `Callback` instances.
    This object wraps a list of `Callback` instances, making it possible
    to call them all at once via a single endpoint
    (e.g. `callback_list.on_gen_end(...)`)."""
    

    def __init__(self, callbacks : list[Callback] | None = None):
        
        self.callbacks = callbacks
        if callbacks is None:
            self.callbacks = []
        
        

        # self.on_simulation_init_callbacks = [callback for callback in self.callbacks if 'on_simulation_init' in dir(callback)]

        # self.on_step_end_callbacks = [callback for callback in self.callbacks if 'on_step_end' in dir(callback)]
        
        # self.on_gen_end_callbacks = [callback for callback in self.callbacks if 'on_gen_end' in dir(callback)]
        # self.on_gen_begin_callbacks = [callback for callback in self.callbacks if 'on_gen_begin' in dir(callback)]
        
        # self.on_run_end_callbacks = [callback for callback in self.callbacks if 'on_run_begin' in dir(callback)]
        
        # self.all_callbacks = [self.on_simulation_init_callbacks, self.on_step_end_callbacks, self.on_gen_begin_callbacks, self.on_gen_end_callbacks, self.on_run_end_callbacks]
        
        self.callbacks_dict : dict[str, list] = dict()

        for name in Callback.callback_points:
            self.callbacks_dict[name] = []
        
            for cbk in self.callbacks:
                if name in dir(cbk):
                    self.callbacks_dict[name].append(cbk)

    def __getattr__(self, name):

        try:
            callbacks = self.callbacks_dict[name]
        except KeyError:
            raise AttributeError(f"{name} is not a valid callback entry point")

        def inner(*args, **kwargs):
            for cbk in callbacks:
                getattr(cbk, name)(*args, **kwargs)

        return inner 
    
    def remove_logging_callbacks(self):
        for name, callbacks in self.callbacks_dict.items():
            for cbk in callbacks: 
                if isinstance(cbk, LoggingCallback):
                    callbacks.remove(cbk)
                    # self.callbacks_dict[name].remove(cbk)
    
    def append(self, cbk : Callback):
        self.callbacks.append(cbk)
        for name in Callback.callback_points:
            if name in dir(cbk):
                self.callbacks_dict[name].append(cbk)

    def __add__(self, clist2):

        for cbk in clist2:
            self.append(cbk)

        return self



    # def on_simulation_init(self, gen : int, step : int, world : dict):
        
    #     for cbk in self.on_simulation_init_callbacks:
            
    #         cbk.on_simulation_init(gen, step, world)
        
    
    # def on_step_end(self, gen : int, step : int, world : dict):
        
    #     for cbk in self.on_step_end_callbacks:
            
    #         cbk.on_step_end(gen, step, world)
        
    
    # def on_gen_end(self, gen, world ):
        
    #     for cbk in self.on_gen_end_callbacks:
            
    #         cbk.on_gen_end(gen, world)
            
    # def on_gen_begin(self, gen, world ):
        
    #     for cbk in self.on_gen_begin_callbacks:
            
    #         cbk.on_gen_begin(gen, world)
            
    # def on_run_end(self, gen, world):
        
        
    #     for cbk in self.on_run_end_callbacks:
         
    #         cbk.on_run_end(gen, world)
       
       
class TimingCallback(Callback):
    pass
   
class TimeGens(TimingCallback):
    
    def __init__(self, print_time_on_gen_end : bool = False, printing_intervall : int = 5) -> None:
        super().__init__()
        
        self.print_time_on_gen_end : bool = print_time_on_gen_end
        self.time_first_gen_start : float | None = None
        self.gen_end_timestamps : list[float] = []
        self.printing_intervall : int = printing_intervall
        
    def on_gen_begin(self, world):
        
        gen = world.current_gen

        if gen == 1:
            
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
            self.log_time()
        

    def get_timing_summary(self):
        total_gens = len(self.gen_end_timestamps) - 1 # -1 because of first timestamp
        total_time = self.gen_end_timestamps[-1] - self.gen_end_timestamps[0]

        timing_dict = {"total gens": total_gens, 
                       "total time": total_time,  
                       "average secs per gen": total_time / total_gens} 
        return timing_dict
            
    def log_time(self):

        data_dict = self.get_timing_summary()
        with open(f"performance/timing{data_dict['total gens']}.json", "w") as fp:
            json.dump(data_dict, fp, indent=4) 
    
    def print_summary(self):
        
        data = self.get_timing_summary()
        summary = data
        print(summary)

            
        
        
        
             

class LoggingCallback(Callback):
    def __init__(self, log_points : str | list[int], n_max_gen : int, n_logs : int):
        
        
        if log_points == 'linear':
            
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

class LogWorldState(LoggingCallback):
    
    def __init__(self, log_points, n_max_gen, n_logs = 5, log_pos=False):
        
        super().__init__(log_points, n_max_gen, n_logs)
        
        self.log_pos = log_pos
        self.multi_species = False 

       
                     
    def on_step_end(self, world):

        gen = world.current_gen
        step = world.current_step
        
        if gen in self.log_points:
            
            world_state = world.world_state
            
            
            
            if step == 1:
                if not os.path.isdir(f"generations/gen{gen}"):
                    os.mkdir(f"generations/gen{gen}")
                if world.n_species > 1:
                    self.multi_species = True 
                    os.mkdir(f'generations/gen{gen}/step_pop_pos_species') 
                else:
                    os.mkdir(f'generations/gen{gen}/step_world_state')
                os.mkdir(f'generations/gen{gen}/step_pop_pos')
                
            
                
                
            
            
            if self.multi_species:
                pop_pos = world.pop_pos
                species = np.array(world.species_list)
                
                pop_pos_species = np.c_[pop_pos, species] 

                np.savetxt(f'generations/gen{gen}/step_pop_pos_species/step{step}', pop_pos_species, fmt="%5i")
                
            else: 
                np.savetxt(f'generations/gen{gen}/step_world_state/step{step}', world_state, fmt="%5i")
                
                
                
            if self.log_pos:
                pop_pos = world.pop_pos
                np.savetxt(f'generations/gen{gen}/step_pop_pos/step{step}', pop_pos, fmt="%5i")
                
            
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

            
            
             
            
class LogWeights(LoggingCallback): # world: flattened nn weights of all dots in in one 2d array
    
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
            
            
            