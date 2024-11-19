import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from dot import Dot
from utils import colors_rgb
#from IPython import display

plt.ion()
mpl.use("Qt5agg") # Qt (PyQt5) backend instead of Tk to stop window from poping to the front 

def downsample_array(arr, max_length=20):
    if len(arr) <= max_length:
        return arr 

    step = len(arr) // max_length  # Determine step size
    downsampled = arr[::step]  # Take every `step`-th element
    downsampled[-1] = arr[-1]
    
    return downsampled

def plot_downsampled(y, *args, **kwargs):

    y_downsampled = downsample_array(y, max_length=300)

    plt.plot(y_downsampled, *args, **kwargs)

def plot(last_gen, survivors, killed, plot_dict):
    #display.clear_output(wait=True)
    
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Generations')
    plt.ylabel('Percentage')
    for i in range(survivors.shape[1]):
        plot_downsampled(survivors[:, i], label=f"survivors {i}", color=tuple(colors_rgb[i, :]))
        plt.text(survivors.shape[0] - 1, survivors[-1, i], str(round(survivors[-1, i], 2)))
    for i in range(killed.shape[1]):
        plot_downsampled(killed[:, i], label=f"killed {i}", color=tuple(colors_rgb[i, :]) + (0.5,))
        plt.text(len(killed[:, i])-1, killed[-1, i], str(killed[-1, i]))
    
    if plot_dict is not None:
        for val_list, name in zip(plot_dict.values(), plot_dict.keys()):
            plot_downsampled(val_list, label=name)
        
    plt.ylim(ymin=0)
    plt.legend(loc="upper left")
    if not last_gen:
        plt.show(block=False)
    else:
        plt.savefig("survivor_killed_plot.png")
        plt.show(block=True)
    plt.pause(.005)
    
    #if last_gen:
        
class PlottingStatCollector:
    pass
        


class StatCollector(PlottingStatCollector):
    
    def __init__(self):
        self.reset()

    def __call__(self, dot, survived):
        self._killed_counts += not dot.alive
        self._survived_counts += survived

    def reset(self):
        self._survived_counts = 0
        self._killed_counts = 0

    @property
    def survived_counts(self):
        return np.array([self._survived_counts])

    @property
    def killed_counts(self):
        return np.array([self._killed_counts])
    

class StatCollectorMultiSpecies(PlottingStatCollector):
    
    def __init__(self, n_species):
        self.n_species = n_species
        self.reset()

    def __call__(self, dot, survived):
        self.killed_counts[dot.species - 1] += not dot.alive
        self.survived_counts[dot.species - 1] += survived

    def reset(self):
        self.survived_counts = np.zeros((self.n_species,), np.uint8)
        self.killed_counts = np.zeros((self.n_species,), np.uint8)




if __name__ == '__main__':

    sc = StatCollectorMultiSpecies(2)

    sc(Dot(1, species=1), True)
    sc(Dot(1, species=2), False)

    print(sc.survived_counts)

