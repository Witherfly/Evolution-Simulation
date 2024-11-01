import numpy as np
import genome_to_color
import cv2
import os 
import json 

os.chdir('logs/')

def newest(dir_path=os.getcwd()):
     files = os.listdir(dir_path)
    
     os.chdir(dir_path)
     
     file = os.path.join(dir_path, max(files, key=os.path.getctime))
     
     os.chdir('..')
     
     return file

def oldest(dir_path=os.getcwd()):
     files = os.listdir(dir_path)
     os.chdir(dir_path)
     
     file = os.path.join(dir_path, min(files, key=os.path.getctime))
     
     os.chdir('..')
     
     return file

os.chdir(newest())

with open('world_configurations/world_params.json') as json_file:
     configurations = json.load(json_file)


gen = 1000                                                            
world_shape = configurations["world_shape"]


death_mask = np.loadtxt('world_configurations/death_mask')
try:
     wall_mask = np.loadtxt('world_configurations/wall_mask')
except FileNotFoundError: 
     wall_mask = np.zeros(world_shape)



reducer = genome_to_color.Genome_to_color('rgb', 6, 3, 4)
img_shape = world_shape + [reducer.color_dim]

first_logged_gen = oldest('generations/')

first_logged_gen_weights = np.loadtxt(os.path.join(first_logged_gen, r'weights\all_weights'))

reducer.fit(first_logged_gen_weights)

death_mask_3d = np.tile(np.atleast_3d(death_mask), (1,1,3)).astype(bool)
wall_mask_3d = np.tile(np.atleast_3d(wall_mask), (1,1,3)).astype(bool)

word_state_colored = np.zeros(img_shape)  
word_state_colored[death_mask_3d] = 0.1  # np.array([26, 26, 26]) # broadcast is to mask 3d array with 2d mask 
word_state_colored[wall_mask_3d] = 0.2    # np.array([140, 140, 140])


pop_pos_dir = f'generations/gen{gen}/step_pop_pos'



images = []

pop_weights = np.loadtxt(f'generations/gen{gen}/weights/all_weights')
os.chdir(pop_pos_dir)

dot_colors = reducer.transform(pop_weights)

for i in range(1, len(os.listdir()) + 1) :
    
    
     pos_array = np.loadtxt(f'step{i}').astype(np.int16)
                  
     world_state_colored_new = word_state_colored.copy() # world_shape is list so shape is (h, w, 3)
     for pos, color in zip(pos_array, dot_colors):
          
          world_state_colored_new[pos[0], pos[1], :] = color 
          
     images.append(world_state_colored_new.copy())
          
          
          

images_resized = [cv2.resize(img, (600, 600)) for img in images]            
         
         
for img in images_resized:
     
     cv2.imshow(f'gen{gen}', img)
     
     cv2.waitKey(0)    
                 
cv2.destroyAllWindows() 