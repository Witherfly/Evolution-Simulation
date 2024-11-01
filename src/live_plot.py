import matplotlib.pyplot as plt
import matplotlib as mpl
#from IPython import display

plt.ion()
mpl.use("Qt5agg") # Qt (PyQt5) backend instead of Tk to stop window from poping to the front 

def plot(last_gen, survivors, killed, plot_dict):
    #display.clear_output(wait=True)
    
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Generations')
    plt.ylabel('Percentage')
    plt.plot(survivors, label="survivors")
    plt.plot(killed, label="killed")
    
    if plot_dict is not None:
        for val_list, name in zip(plot_dict.values(), plot_dict.keys()):
            plt.plot(val_list, label=name)
        
    plt.ylim(ymin=0)
    plt.legend(loc="upper left")
    plt.text(len(survivors)-1, survivors[-1], str('{:.2f}'.format(survivors[-1])))
    plt.text(len(killed)-1, killed[-1], str(killed[-1]))
    if not last_gen:
        plt.show(block=False)
    else:
        plt.savefig("survivor_killed_plot.png")
        plt.show(block=True)
    plt.pause(.005)
    
    #if last_gen:
        
        
        