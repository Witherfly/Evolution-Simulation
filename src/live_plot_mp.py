
import multiprocessing as mp 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import sleep

from utils import colors_rgb


class PlotterMP:
    def __init__(self, kill_enabled=True):
        
        self.survived = []
        self.killed = []
        self.len = 0
        
    
    def stop(self):
        # plt.savefig("survivor_killed_plot_mp.png")
        plt.close('all')
        print('terminated plotter')

    def process_new_data(self, data):

        self.len += 1
        self.survived.append(data['survived'])
        try:
            self.killed.append(data['killed'])
        except KeyError:
            return np.array(self.survived), np.empty((0, 0))
        
        return np.array(self.survived), np.array(self.killed)


    def plot(self):
        
        survived = None
        while not self.queue.empty():
            data = self.queue.get()

            if data is None:
                self.stop()
                return False

            survived, killed = self.process_new_data(data)
        
        if survived is not None:
            self.ax.clear()
            for i in range(survived.shape[1]):
                self.ax.plot(survived[:, i], label=f"survivors {i+1}", color=tuple(colors_rgb[i, :]))
                plt.text(survived.shape[0] - 1, survived[-1, i], str(round(survived[-1, i], 2)))
            
            for i in range(killed.shape[1]):
                self.ax.plot(killed[:, i], label=f"killed {i+1}", color=tuple(colors_rgb[i, :]) + (0.5,))
                plt.text(len(killed[:, i])-1, killed[-1, i], str(killed[-1, i]))


            plt.legend(loc="upper left")

            self.fig.canvas.draw()
        return True

    def init_plt(self):
        plt.title('Training...')
        plt.xlabel('Number of Generations')
        plt.ylabel('Percentage')
        plt.ylim(ymin=0)

    def start(self, queue : mp.Queue):
        # plt.ion()
        print('...starting plotter')
        self.queue = queue
        self.init_plt()
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=200)
        timer.add_callback(self.plot)
        timer.start()

        print('...done')
        plt.show()
        print('dondone')


class DataSource:

    def __init__(self):

        self.plotter = PlotterMP()
        self.queue = mp.Queue()
        self.plot_process = mp.Process(target=self.plotter.start, args=(self.queue,), daemon=True)
        self.plot_process.start() 

    def start(self):

        mpl.use("Qt5agg") # Qt (PyQt5) backend instead of Tk to stop window from poping to the front 
        lin_space = np.linspace(0, np.pi*4, 100)
        lin_space = np.vstack([lin_space, lin_space]).T

        sleep(1)

        for i in range(len(lin_space)):

            data1 = np.sin(lin_space[i])
            data1[1] += 0.1
            data2 = np.cos(lin_space[i])
            data2[1] += 0.1
            data = {'survived' : data1, 'killed' : data2}
            self.queue.put(data)
            sleep(0.06)
        
        sleep(3)
        self.queue.put(None)
        self.plot_process.kill()




if __name__ == '__main__':
    d = DataSource()
    d.start()