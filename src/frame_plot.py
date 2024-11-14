import numpy as np
import matplotlib.pyplot as plt
import cv2 
from utils import timing

# Sample function to generate plotting points
def generate_points(n, a=1.0):
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(a*x)
    return x, y

# Function to create and save frames as numpy arrays
def plot_framed(x, Y, Y_labels):
    frames = []
    
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], '-', label=Y_labels[i])[0] for i in range(len(Y))]
    ax.set_xlim(0, x.max() *1.02)
    y_max = max(y.max() for y in Y)
    ax.set_ylim(-1, y_max *1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc = "upper left")
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Set up blitting
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

     # Initialize crosshair lines and text
    crosshair_h, = ax.plot([], [], 'k--', linewidth=0.5)  # Horizontal line
    crosshair_v, = ax.plot([], [], 'k--', linewidth=0.5)  # Vertical line
    point_text = ax.text(0, 0, '', fontsize=12, color='black', ha='left')
    
    for i in range(len(x)):
        fig.canvas.restore_region(background)  # Restore background
        
        # Update each line with the current frame data
        for line, y in zip(lines, Y):
            line.set_data(x[:i+1], y[:i+1])
            ax.draw_artist(line)  # Draw each updated line       # Update crosshair lines and text at the latest point

        
        crosshair_h.set_data([ax.get_xlim()[0], x[i]], [Y[0][i], Y[0][i]])
        crosshair_v.set_data([x[i], x[i]], [ax.get_ylim()[0], Y[0][i]])
        point_text.set_position((x[i], Y[0][i]*1.01))
        point_text.set_text(f'{round(Y[0][i], 2)}')
        
        # Draw updated elements
        ax.draw_artist(crosshair_h)
        ax.draw_artist(crosshair_v)
        ax.draw_artist(point_text)
        # Update line data progressively
        
        fig.canvas.blit(ax.bbox)
        
        # Convert the canvas to a numpy array and store it
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame.copy())
    
    plt.close(fig)  # Close the figure after saving all frames
    return frames



if __name__ == '__main__':
    # Generate points and save frames
    n = 100
    x, y1 = generate_points(n)
    x, y2 = generate_points(n, a=4)
    save_frames_as_arrays = timing(save_frames_as_arrays)
    frames = save_frames_as_arrays(x, [y1, y2])

    # exit()
    i = 0
    i_max = len(frames) - 1
    while True:
        # if i < 0:
        #     i = i_max 
        
        cv2.imshow('name', frames[i])

        if cv2.waitKeyEx() == 2424832:
            i -= 1
        else: 
            i += 1

        if i > i_max:
            break
    
    cv2.destroyAllWindows()
    # frames now contains each frame as a numpy array
