import sys
PY311 = sys.version_info >= (3,11)
from typing import Tuple
if PY311:
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import rasterio


class ArrayProfile(object):
    """
    A class to define array attribute. This class aims to be used to mock
    a numpy array object in order to access its attributes members such as
    ndim, shape, dtype and size.
    """
    def __init__(self, shape: Tuple[int], ndim: int, dtype: np.dtype):
        """
        Constructor
        
        Args:
            shape: array shape
            ndim: array number of dimensions
            dtype: array data type
        """
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(self.shape)
        
    @classmethod
    def from_dataset(cls, ds: rasterio.io.DatasetReader) -> Self:
        """
        Method to create an object from a rasterio Dataset.
        
        Args:
            ds: a rasterio dataset reader object (from rasterio.open)
            
        Returns:
            the instanciated ArrayProfile object
        """
        shape = (ds.count, ds.height, ds.width)
        ndim = 3
        if ds.count == 1:
            shape = (ds.height, ds.width)
            ndim = 2
        return cls(
                shape=shape,
                ndim=ndim,
                dtype=np.dtype(ds.profile['dtype']))
