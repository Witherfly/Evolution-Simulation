from typing import Iterable, Tuple
import colorsys
import itertools
from fractions import Fraction
from pprint import pprint

import cv2 as cv 
import numpy as np

def zenos_dichotomy() -> Iterable[Fraction]:
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1,2**k)

def fracs() -> Iterable[Fraction]:
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield Fraction(0)
    for k in zenos_dichotomy():
        i = k.denominator # [1,2,4,8,16,...]
        for j in range(1,i,2):
            yield Fraction(j,i)

# can be used for the v in hsv to map linear values 0..1 to something that looks equidistant
# bias = lambda x: (math.sqrt(x/3)/Fraction(2,3)+Fraction(1,3))/Fraction(6,5)

HSVTuple = Tuple[Fraction, Fraction, Fraction]
RGBTuple = Tuple[float, float, float]

def hue_to_tones(h: Fraction) -> Iterable[HSVTuple]:
    for s in [Fraction(6,10)]: # optionally use range
        for v in [Fraction(8,10),Fraction(5,10)]: # could use range too
            yield (h, s, v) # use bias for v here if you use range

def hsv_to_rgb(x: HSVTuple) -> RGBTuple:
    return colorsys.hsv_to_rgb(*map(float, x))

flatten = itertools.chain.from_iterable

def hsvs() -> Iterable[HSVTuple]:
    return flatten(map(hue_to_tones, fracs()))

def rgbs() -> Iterable[RGBTuple]:
    return map(hsv_to_rgb, hsvs())

def rgb_to_css(x: RGBTuple) -> str:
    uint8tuple = map(lambda y: int(y*255), x)
    #return "rgb({},{},{})".format(*uint8tuple)
    return list(uint8tuple)

def css_colors() -> Iterable[str]:
    return map(rgb_to_css, rgbs())

def distinct_colors(n_colors : int) -> np.array:
    
    return np.array(list(itertools.islice(css_colors(), n_colors)))

if __name__ == "__main__":
    # sample 100 colors in css format
    sample_colors = list(itertools.islice(css_colors(), 10))

    #pprint(sample_colors)
    colors = np.tile(np.rot90(np.array(sample_colors).reshape(-1, 3, 1), axes=(1, 2)), (1, 2, 1)).astype(np.uint8)
    print(colors.shape)
    
    cv.imshow('colors', cv.resize(colors, (2000, 100)))
    cv.waitKey(0)
    
    