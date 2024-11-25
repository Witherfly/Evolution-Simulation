import os 
import sys
import shutil
import pathlib

from utils import get_number_at_end

def remove_gens(remove_unit_gen : int = 1):

    os.chdir('logs/')
    for dir in os.listdir():
        generations = os.listdir(os.path.join(dir, 'generations'))
        for gen in generations:
            if get_number_at_end(gen) <= remove_unit_gen:
                print(dir)
                


def remove_empty_runs(remove_below_n_logged_gens : int = 0):

    os.chdir('logs/')
    n_removed = 0
    for dir in os.listdir():
        generations = os.listdir(os.path.join(dir, 'generations'))
        if len(generations) <= remove_below_n_logged_gens:
            shutil.rmtree(dir)
            n_removed += 1
    
    print(f'{n_removed} files removed')

if __name__ == '__main__':
    try:
        remove_min_gen = int(sys.argv[1])
    except IndexError:
        remove_min_gen = 0
    remove_empty_runs(remove_min_gen)
    # remove_gens(remove_min_gen)

