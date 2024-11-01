import datetime
import cv2
import numpy as np

world_shape = (50, 50)

# init a canvas
canvas = np.zeros(np.array(world_shape) * 10, np.uint8)

# make canvas white


# global coordinates and drawing state
x = 0
y = 0
drawing_zone = False
drawing_wall = False 
drawing_line = False 

# mouse callback function
def draw(event, current_x, current_y, flags, params):
    # hook up global variables
    global x, y, drawing_zone, drawing_wall
    
    # handle mouse down event
    if event == cv2.EVENT_LBUTTONDOWN:
        # update coordinates
        x = current_x
        y = current_y
        
        # enable drawing flag
        drawing_zone = True
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        # update coordinates
        x = current_x
        y = current_y
        
        # enable drawing flag
        drawing_wall = True 
    
    
    # handle mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        # draw only if mouse is down
        if drawing_zone:
            # draw the line
            cv2.line(canvas, (current_x, current_y), (x, y), 122, thickness=50)
            
            # update coordinates
            x, y = current_x, current_y
            
        if drawing_wall:
            # draw the line
            cv2.line(canvas, (current_x, current_y), (x, y), 255, thickness=10)
            
            # update coordinates
            x, y = current_x, current_y
    
    # handle mouse up event
    elif event == cv2.EVENT_LBUTTONUP:
        # disable drawing flag
        drawing_zone = False
        
    elif event == cv2.EVENT_RBUTTONUP:
        
        drawing_wall = False 
    


# display the canvas in a window
cv2.imshow('Draw', canvas)

# bind mouse events
cv2.setMouseCallback('Draw', draw)

# infinite drawing loop

time = str(datetime.datetime.now()).split(".")[0].replace(':', ' ')

         
l_press_count = 0 

while True:
    # update canvas
    cv2.imshow('Draw', canvas)
    key = cv2.waitKey(1)
    
    if key & 0xFF == 108: # l key 
        l_press_count += 1
        
        if l_press_count % 2 == 1:
            pass      
    
    # break out of a program on 'Esc' button hit
    if key & 0xFF == 27: 
        
        canvas_resized = cv2.resize(canvas, world_shape)
        
        zone_mask = canvas_resized == 122
        
        wall_mask = canvas_resized == 255 
        
        
        np.savetxt(f"custom_masks/death_masks/mask{time}", zone_mask, fmt="%5i")
        if not np.all(wall_mask == 0 ):
            np.savetxt(f"custom_masks/wall_masks/mask{time}", wall_mask, fmt="%5i")
        break

# clean up windows
cv2.destroyAllWindows()