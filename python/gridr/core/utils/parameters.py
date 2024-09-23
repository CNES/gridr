from typing import Union, Tuple, Any

def tuplify(p: Union[Any, Tuple], ndim: int):
    out = p
    try:
        p[0]
    except TypeError:
        out = ((p, p), ) * ndim
    return out
