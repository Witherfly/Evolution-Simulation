import os 
import datetime

time = str(datetime.datetime.now()).split(".")[0].replace(':', ' ')

current_wdir = os.getcwd()

dir_path = os.path.join(current_wdir, 'src', 'logs', f'run_{time}')

os.mkdir(dir_path)
    
os.chdir(dir_path) 

os.mkdir("world_configurations")
os.mkdir("generations")
os.mkdir("colorizing_data")
os.mkdir("performance")

np.savetxt("world_configurations/death_mask", self.death_func.mask, fmt="%5i")
if self.obstacles_enababled:
    np.savetxt("world_configurations/wall_mask", self.wall_mask, fmt="%5i")

with open("world_configurations/world_params.json", "w") as fp:
    param_dict = self.get_config()
    json.dump(param_dict, fp, indent=4 ) 