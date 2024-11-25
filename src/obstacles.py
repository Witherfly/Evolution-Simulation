import numpy as np
import cv2
import datetime
import os 

import numpy.typing as npt

from utils import get_newest_file

class Wall():
    
    def __init__(self,world_shape : tuple[int, int]):
        
        self.world_shape = world_shape
        
        self.wall_mask = np.zeros(self.world_shape)
        self.mask = self.create_halfway_wall()
    
    
    def create_halfway_wall(self) -> npt.NDArray[np.bool_]:
        
        h, w = self.world_shape
        mask = np.zeros(self.world_shape, dtype=np.bool_)
        
        wall_idx_east = int(w * 0.1)
        wall_idx_west = int(w * (1 - 0.1))  
        
        walls_height = h * 0.4
        
        north_idx = int((h - walls_height) / 2)
        south_idx = h - north_idx + 1
        
        mask[north_idx:south_idx, [wall_idx_east, wall_idx_west]] = True 
        
        return mask
        
class Load_custom_wall():
    
    def __init__(self, file_name, world_shape) -> None:
        
        dir_path = os.path.join(os.getcwd(), "custom_masks", "wall_masks")
        
        
        if file_name == "newest":
            
            full_path = get_newest_file(dir_path, return_full_path=True)
            
        
        self.mask : npt.NDArray[np.bool_] = np.loadtxt(full_path, dtype=np.bool_)  
        
        if self.mask.shape != world_shape: 

            self.mask = cv2.resize(self.mask, world_shape)

if __name__ == '__main__':
    
    world_shape = (50, 50)
    wall1 = Wall(world_shape)
    mask = wall1.mask
    
    time = str(datetime.datetime.now()).split(".")[0].replace(':', ' ')
    
    np.savetxt(f"test_wall_masks/wall_mask{time}", mask)