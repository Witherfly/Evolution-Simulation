import os 
import re 
from contextlib import contextmanager
import numpy as np
import cv2


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


def get_number_at_end(input_string : str) -> int | None:
    match = re.search(r'(\d+)$', input_string)
    return int(match.group()) if match else None



colors_rgb = np.array([[200, 0, 0],
                    [0, 200, 0],
                    [0, 0, 200],
                    [ 51, 127, 127],
                    [142, 204,  81]], dtype=np.float32) / 256

text_specs = {  "fontFace"                   : cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale"              : 1,
                "color"              : (0,0,0),
                "thickness"              : 1,
                "lineType"               : 2,}

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)