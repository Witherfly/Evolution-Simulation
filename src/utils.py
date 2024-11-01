import os 
from contextlib import contextmanager


@contextmanager
def remember_cwd():
    curdir = os.getcwd()
    try: 
        yield
    finally: 
        os.chdir(curdir)

def get_newest_file(dir_path=None):
    with remember_cwd():
        if dir_path is not None:
            os.chdir(dir_path)
        files = os.listdir()

        max_file = max(files, key=os.path.getctime)
    
    return max_file