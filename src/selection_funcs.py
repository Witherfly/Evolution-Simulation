import numpy as np
import datetime
import os

from .utils import get_newest_file

class Zone_selection:
    def __init__(self, zone_type : str | tuple[str, ...], alive_area : float | tuple[float, ...], world_shape : tuple[int, int]):
        
        self.alive_area = alive_area
        self.world_shape = world_shape 
        self.zone_type = zone_type
        
        if isinstance(zone_type, str):
            self.multi_mask = False 
        else:
            self.multi_mask = True
        
        
        
        mask_func_dict = {"circle": self.create_circular_mask, "east_west": self.east_west_mask, "north_south": self.north_south_mask}
        
        
        
        if self.multi_mask:
                 
            self.mask_list = [mask_func_dict[name](alive_a) for name, alive_a in zip(self.zone_type, self.alive_area)]
 # type: ignore            
            # create main mask
            self.mask = np.zeros(self.world_shape)
            for idx, mask in enumerate(self.mask_list):
                
                self.mask = np.logical_or(self.mask, mask)
                

        else:
            self.mask = mask_func_dict[zone_type](self.alive_area)
        
            
            
    def east_west_mask(self, alive_area : float) -> np.ndarray:
        
        h, w = self.world_shape
        
        xx, yy = np.meshgrid(np.arange(h), np.arange(w))
        
        west_boundary = self.world_shape[1] * alive_area * 0.5
        east_boundary = self.world_shape[1] * (1 - alive_area * 0.5) - 1
        mask = np.logical_or(xx < west_boundary, xx > east_boundary)
        
        return mask  
    
    def north_south_mask(self, alive_area : float):
        
        h, w = self.world_shape
        
        xx, yy = np.meshgrid(np.arange(h), np.arange(w))
        
        north_boundary = self.world_shape[0] * alive_area * 0.5
        south_boundary = self.world_shape[0] * (1 - alive_area * 0.5) - 1
        mask = np.logical_or(yy < north_boundary, yy > south_boundary)
        
        return mask  
        
    def create_circular_mask(self, alive_area : float):
        
        h, w = self.world_shape
        world_size = h * w
        
        if isinstance(alive_area, int):
            radius = np.sqrt(alive_area / np.pi)
        else:
            radius = np.sqrt((world_size * alive_area)/ np.pi )
        center = None 
        
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask
        
    def __call__(self, pop_pos : np.ndarray) -> tuple[np.ndarray, dict[str, int] | dict]:
        
        zone_info = {}
            
        if self.multi_mask:
            
            
            
            for mask, name in zip(self.mask_list, self.zone_type):
                
                is_inside = mask[pop_pos[:, 0], pop_pos[:, 1]]
                zone_info.update({name: np.sum(is_inside)})
            
        is_alive = self.mask[pop_pos[:, 0], pop_pos[:, 1]]
        

        return is_alive, zone_info 

class Load_custom():
    
    def __init__(self, dir_path : str, file_name : str):
        
        dir_path = os.path.join("src\custom_masks", dir_path)
        
        
        if file_name == "newest":
            
            file_name = get_newest_file(dir_path)
            
        full_path = os.path.join(dir_path, file_name)
        
        
        self.mask = np.loadtxt(full_path)
          
    def __call__(self, pop_pos : np.ndarray) -> tuple[np.ndarray, None]:
        
        is_alive = self.mask[pop_pos[:, 0], pop_pos[:, 1]]
        
        zone_info = None
        return is_alive, zone_info

if __name__ == '__main__':
    
    world_shape = (50, 50)
    func1 = Zone_selection(("circle", "east_west"), (0.15, 0.06), world_shape)

    mask = func1.mask
    
    time = str(datetime.datetime.now()).split(".")[0].replace(':', ' ')
    
    np.savetxt(f"test_masks/selection_mask{time}", mask)