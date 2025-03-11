# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Shared Memory Utils module
"""
from datetime import datetime
from functools import wraps
from multiprocessing import shared_memory
from typing import List
from uuid import uuid4

import numpy as np


class SharedMemoryArray(object):
    COUNTER=0
    
    """A class handler for shared_memory buffer and associated numpy array
    """
    def __init__(self, shape, dtype, name, array_slice=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.array_slice = array_slice
        self.smh = None
        self.array = None
    
    def create(self):
        """Create the shared_memory buffer corresponding to object's attributes.
        Please note that the slice is not taking into account here - it is used
        in case of load.
        """
        size = np.dtype(self.dtype).itemsize * np.prod(self.shape)
        self.smh = shared_memory.SharedMemory(create=True, size=size,
                name=self.name)
        self.array = np.ndarray(shape=self.shape, dtype=self.dtype,
                buffer=self.smh.buf)
        
    def load(self):
        """Load the object from a previously created buffer corresponding to 
        the current object's name attribute.
        """
        # Reconnect to the Shared Memory buffer
        self.smh = shared_memory.SharedMemory(name=self.name)
        self.array = np.ndarray(self.shape, dtype=self.dtype,
                buffer=self.smh.buf)
        if self.array_slice:
            self.array = self.array[self.array_slice]
    
    def close(self):
        """
        """
        self.smh.close()
        self.smh = None
        self.array = None
        
    @classmethod
    def clone(cls, sma, **override):
        kwargs = {'shape': sma.shape, 'dtype': sma.dtype, 'name': sma.name,
                'array_slice': sma.array_slice}
        kwargs.update(override)
        return cls(**kwargs)
    
    @classmethod
    def build_sma_name(cls, prefix: str) -> str:
        """Generate a supposed unique sma_name from timestamp and uuid4.
        This methods also uses and increments a class variable counter.
        """
        if prefix is None:
            prefix=''
        cls.COUNTER += 1
        sma_name = "-".join((str(cls.COUNTER), prefix,
                datetime.now().strftime('%Y%m-%d%H-%M%S'), str(uuid4())))
        return sma_name
    
    @classmethod
    def clear_buffers(cls, buffer_names: List[str]):
        """Clear a list of buffer's names
        
        Args:
            buffer_names: a list of buffer's names
        """
        for name in buffer_names:
            buf = shared_memory.SharedMemory(name=name)
            buf.close()
            buf.unlink()

def shmarray_wrap(func):
    """A decorator to auto detect arguments that are passed as SharedMemoryArray
    and replace them by the corresponding loaded array.
    That's a helper function to be used with caution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        """
        smas = []
        def resolve_arg(arg):
            ret = arg
            if isinstance(arg, SharedMemoryArray):
                smas.append(arg)
                arg.load()
                ret = arg.array
            return ret
        func_ret = None
        res_args = [resolve_arg(arg) for arg in args]
        res_kwargs = {key:resolve_arg(arg) for key, arg in kwargs.items()}
        try:
            func_ret = func(*res_args, **res_kwargs)
        except:
            raise
        finally:
            for sma in smas:
                sma.close()
        return func_ret
    return wrapper

def create_and_register_sma(
        shape, dtype,
        register: List[str],
        prefix:str = None,
        ) -> SharedMemoryArray:
    """Helper method to create and register (save in a list) a SharedMemoryArray
    """
    buffer_name = SharedMemoryArray.build_sma_name(prefix)
    buffer = SharedMemoryArray(shape=shape, dtype=dtype, name=buffer_name)
    buffer.create()
    register.append(buffer_name)
    return buffer
    