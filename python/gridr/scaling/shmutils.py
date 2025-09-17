# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://github.com/CNES/gridr).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Shared Memory Utils module
"""
from datetime import datetime
from functools import wraps
from multiprocessing import shared_memory
from typing import List, NoReturn, Optional, Tuple
from uuid import uuid4

import numpy as np


class SharedMemoryArray(object):
    """
    A class handler for managing shared memory buffers and their associated
    NumPy array views.

    This class simplifies the creation, loading, and cleanup of NumPy arrays
    backed by Python's `multiprocessing.shared_memory` module, enabling
    efficient data sharing between processes.

    Attributes
    ----------
    COUNTER : int
        A class-level counter used for generating unique shared memory names.

    shape : tuple of int
        The shape of the NumPy array that will reside in shared memory.

    dtype : numpy.dtype
        The data type of the NumPy array elements.

    name : str
        The unique name of the shared memory segment.

    array_slice : tuple of slice, optional
        A tuple of slice objects to apply to the array after loading, providing
        a view of a sub-section of the shared memory.

    smh : shared_memory.SharedMemory or None
        The handler for the shared memory buffer. `None` if not yet created or
        loaded, or after `close()`.

    array : numpy.ndarray or None
        The NumPy array view onto the shared memory buffer. `None` if not yet
        created or loaded, or after `close()`.
    """

    COUNTER = 0

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        name: str,
        array_slice: Optional[Tuple[slice, ...]] = None,
    ):
        """
        Initializes a SharedMemoryArray instance.

        Parameters
        ----------
        shape : tuple of int
            The desired shape of the NumPy array.

        dtype : numpy.dtype
            The desired data type of the NumPy array.

        name : str
            A unique name for the shared memory segment. This name is used
            to create or connect to the shared memory.

        array_slice : tuple of slice, optional
            A tuple of slice objects (e.g., `(slice(0, 10), slice(None))`)
            to apply to the NumPy array after it is loaded from the shared
            memory buffer. This allows working with a subset of the shared
            memory. Defaults to `None`.
        """
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.array_slice = array_slice
        self.smh = None
        self.array = None

    def create(self) -> NoReturn:
        """
        Creates the shared memory buffer and associates a NumPy array view.

        This method allocates a shared memory segment with the specified `name`
        and `size` (derived from `shape` and `dtype`), then creates a NumPy
        array view that points to this shared memory. The `array_slice`
        attribute is not applied during creation; it's used when the array is
        loaded (e.g., by another process, or via the `load()` method).
        """
        size = np.dtype(self.dtype).itemsize * np.prod(self.shape)
        self.smh = shared_memory.SharedMemory(create=True, size=size, name=self.name)
        self.array = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self.smh.buf)

    def load(self) -> NoReturn:
        """
        Loads the object by connecting to a previously created shared memory
        buffer.

        This method attempts to connect to an existing shared memory segment
        identified by `self.name`. It then creates a NumPy array view onto
        this shared memory. If `self.array_slice` is defined, it applies this
        slice to the array, providing a view to a specific region.
        """
        # Reconnect to the Shared Memory buffer
        self.smh = shared_memory.SharedMemory(name=self.name)
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.smh.buf)
        if self.array_slice:
            self.array = self.array[self.array_slice]

    def close(self) -> NoReturn:
        """
        Closes the connection to the shared memory buffer.

        This method releases the current process's view of the shared memory
        segment. It does *not* unlink (delete) the shared memory segment itself;
        it only closes the current process's connection. After calling
        `close()`, `self.smh` and `self.array` are set to `None`.
        """
        self.smh.close()
        self.smh = None
        self.array = None

    @classmethod
    def clone(
        cls,
        sma: "SharedMemoryArray",
        **override,
    ) -> "SharedMemoryArray":
        """
        Creates a new `SharedMemoryArray` instance by cloning an existing one.

        This class method allows for creating a new `SharedMemoryArray` object
        with attributes copied from another `SharedMemoryArray` instance.
        Specific attributes can be overridden by providing keyword arguments.

        Parameters
        ----------
        sma : SharedMemoryArray
            The `SharedMemoryArray` instance to clone. Its `shape`, `dtype`,
            `name`, and `array_slice` attributes will be used as defaults.

        **override
            Keyword arguments to override any of the cloned attributes.
            For example: `name="new_name"`, `shape=(10, 20)`.

        Returns
        -------
        SharedMemoryArray
            A new `SharedMemoryArray` instance with the cloned or overridden
            attributes.
        """
        kwargs = {
            "shape": sma.shape,
            "dtype": sma.dtype,
            "name": sma.name,
            "array_slice": sma.array_slice,
        }
        kwargs.update(override)
        return cls(**kwargs)

    @classmethod
    def build_sma_name(cls, prefix: str) -> str:
        """
        Generates a supposedly unique name for a shared memory segment.

        The name is constructed using a class-level counter, an optional prefix,
        the current timestamp, and a UUID4 string to maximize uniqueness. The
        class counter is incremented with each call.

        Parameters
        ----------
        prefix : str, optional
            An optional string prefix to include in the generated name.
            Defaults to `None`, resulting in an empty prefix.

        Returns
        -------
        str
            A unique string suitable for use as a shared memory segment name.
            Example:
            "1-my_prefix-202310-2715-3000-abcdef12-3456-7890-abcd-ef1234567890"
        """
        if prefix is None:
            prefix = ""
        cls.COUNTER += 1
        sma_name = "-".join(
            (str(cls.COUNTER), prefix, datetime.now().strftime("%Y%m-%d%H-%M%S"), str(uuid4()))
        )
        return sma_name

    @classmethod
    def clear_buffers(cls, buffer_names: List[str]) -> NoReturn:
        """
        Clears (unlinks) a list of named shared memory buffers.

        This method iterates through a list of shared memory names and attempts
        to unlink (delete) each corresponding shared memory segment from the
        operating system. This effectively cleans up shared memory resources.

        Parameters
        ----------
        buffer_names : list of str
            A list of unique names of the shared memory buffers to be unlinked.

        """
        for name in buffer_names:
            buf = shared_memory.SharedMemory(name=name)
            buf.close()
            buf.unlink()


def shmarray_wrap(func):
    """
    A decorator to automatically load and close `SharedMemoryArray` instances
    passed as arguments to a wrapped function.

    This helper function simplifies working with `SharedMemoryArray` objects
    by automatically handling their `load()` and `close()` operations.
    It's intended for functions that operate on NumPy arrays but might receive
    `SharedMemoryArray` instances as inputs.

    Parameters
    ----------
    func : callable
        The function to be wrapped. Its arguments will be inspected for
        `SharedMemoryArray` instances.

    Returns
    -------
    callable
        A wrapper function that handles the loading and closing of
        `SharedMemoryArray` arguments before and after executing the original
        `func`.

    Notes
    -----
    This decorator should be used with caution as it modifies the arguments
    passed to the wrapped function by replacing `SharedMemoryArray` instances
    with their underlying NumPy arrays. It ensures `close()` is called on
    all detected `SharedMemoryArray` instances, even if the wrapped function
    raises an exception.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function created by the `shmarray_wrap` decorator.

        This function intercepts calls to the decorated function. It iterates
        through both positional and keyword arguments, identifies any
        `SharedMemoryArray` instances, calls their `load()` method to make
        their `array` attribute available, and then passes these `np.ndarray`
        views to the original function.

        It ensures that `close()` is called on all `SharedMemoryArray` instances
        that were loaded, regardless of whether the wrapped function completes
        successfully or raises an exception.

        Parameters
        ----------
        *args
            Positional arguments passed to the decorated function.
        **kwargs
            Keyword arguments passed to the decorated function.

        Returns
        -------
        any
            The return value of the wrapped function.

        Raises
        ------
        Exception
            Any exception raised by the wrapped function will be re-raised
            after ensuring all `SharedMemoryArray` instances are closed.
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
        res_kwargs = {key: resolve_arg(arg) for key, arg in kwargs.items()}
        try:
            func_ret = func(*res_args, **res_kwargs)
        except Exception:
            raise
        finally:
            for sma in smas:
                sma.close()
        return func_ret

    return wrapper


def create_and_register_sma(
    shape,
    dtype,
    register: List[str],
    prefix: str = None,
) -> SharedMemoryArray:
    """Creates a `SharedMemoryArray` and registers its name in a list.

    This helper function simplifies the process of creating a new shared memory
    array. It first generates a unique name for the shared memory segment, then
    instantiates and creates the `SharedMemoryArray` object, and finally adds
    the generated shared memory name to a provided list for tracking (e.g., for
    later cleanup).

    Parameters
    ----------
    shape : tuple of int
        The desired shape of the NumPy array to be created in shared memory.

    dtype : numpy.dtype
        The data type of the NumPy array elements.

    register : list of str
        A list to which the generated unique name of the shared memory segment
        will be appended. This list is typically used to keep track of active
        shared memory buffers for later management or clearing.

    prefix : str, optional
        An optional string prefix to use when generating the unique name for
        the shared memory segment. Defaults to `None`.

    Returns
    -------
    SharedMemoryArray
        The newly created and initialized `SharedMemoryArray` object.

    """
    buffer_name = SharedMemoryArray.build_sma_name(prefix)
    buffer = SharedMemoryArray(shape=shape, dtype=dtype, name=buffer_name)
    buffer.create()
    register.append(buffer_name)
    return buffer
