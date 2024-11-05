


    
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def func(a, b):
    return a + b

def foo(a):
    return a * 2


class MyWorld:
    def __init__(self) -> None:
        self.const = 'a'
        self.l = [1, 2, 3, 4]
        self.n = 3
    
    def create_obs(self, i):
        return [self.l[i]]
    
    def start_sim(self):
        
        for i in range(4):
            with Pool() as pool:
                obs = pool.map(self.create_obs, range(i))

            print(obs)

def main():
    world = MyWorld()

    world.start_sim()

if __name__=="__main__":
    # freeze_support()
    main()