
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
    
    def remove_callbacks_type(self, BaseClass):
        for name, callbacks in self.callbacks_dict.items():
            for cbk in callbacks: 
                if isinstance(cbk, BaseClass):
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
       
       

