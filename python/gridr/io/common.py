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
Module for common IO definitions
# @doc
"""
from enum import IntEnum
from typing import Any, Optional, Type

import rasterio


class GridRIOMode(IntEnum):
    """
    Defines input/output (I/O) modes for computations.

    This enumeration is used to specify whether a particular operation or data
    context pertains to input or output.

    Members
    -------
    INPUT : int
        Represents an input mode (value = 1).
    OUTPUT : int
        Represents an output mode (value = 2).
    """

    INPUT = 1
    OUTPUT = 2


class SafeContext:
    """
    A context manager designed to safely wrap another resource, especially
    useful for optional resources or those whose context manager status is
    uncertain.

    This context manager provides flexible behavior based on the wrapped
    `resource`:

    1.  If `resource` is `None`:
        * `__enter__` will return `None`.
        * `__exit__` will perform no action.

    2.  If `resource` is a valid context manager (i.e., it implements both
        `__enter__` and `__exit__` methods):
        * `SafeContext` will delegate to that resource's `__enter__` and
        `__exit__` methods, effectively behaving just like the wrapped
        resource's own context manager.

    3.  If `resource` is not `None` but also not a context manager:
        * `__enter__` will simply return the `resource` itself.
        * `__exit__` will perform no action.

    This class ensures that operations within a `with` statement are performed
    safely without errors even if the underlying resource is `None` or not
    a proper context manager.

    Examples
    --------
    >>> class MyResource:
    ...     def __init__(self, name):
    ...         self.name = name
    ...         print(f"Resource {self.name} created")
    ...     def close(self):
    ...         print(f"Resource {self.name} closed")
    >>>
    >>> # Case 1: Wrapping a context manager (like a file)
    >>> with SafeContext(open("temp.txt", "w")) as f:
    ...     if f:
    ...         f.write("Hello")
    ... # File 'temp.txt' is automatically closed here.
    >>> import os
    >>> os.remove("temp.txt")
    >>>
    >>> # Case 2: Wrapping None
    >>> with SafeContext(None) as res:
    ...     print(f"Resource inside context: {res}")
    ... # Output: Resource inside context: None
    >>>
    >>> # Case 3: Wrapping a non-context manager object
    >>> with SafeContext(MyResource("Test")) as res:
    ...     if res:
    ...         print(f"Resource name: {res.name}")
    ... # Output: Resource Test created
    ... # Output: Resource name: Test
    >>> # MyResource.close() is NOT called automatically as it's not a context manager
    """

    def __init__(self, resource: Any):
        """
        Initializes the SafeContext with the given resource.

        Parameters
        ----------
        resource : any
            The resource to be managed by this context manager. This can be:

              - `None` (no operation will be performed).
              - An object that implements `__enter__` and `__exit__` methods
                (a context manager, e.g., a file object, a
                `rasterio.DatasetReader`).
              - Any other arbitrary Python object.

        """
        self._resource = resource

    def __enter__(self) -> Any:
        """
        Enters the runtime context.

        This method determines the behavior based on the type of the wrapped
        resource:

        -   If `self._resource` is `None`, it returns `None`.
        -   If `self._resource` has an `__enter__` method, it calls and returns
            the result of `self._resource.__enter__()`.
        -   Otherwise (if `self._resource` is not `None` but not a context
            manager), it returns `self._resource` directly.

        Returns
        -------
        any
            The managed resource (or its `__enter__` return value), or `None` if
            the initial resource was `None`.
        """
        if self._resource is not None:
            # If the resource has an __enter__ method, call it
            if hasattr(self._resource, "__enter__"):
                return self._resource.__enter__()
            else:
                # If it's not None but not a context manager, return it directly
                return self._resource
        return None  # Return None if the initial resource was None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> Optional[bool]:
        """
        Exits the runtime context.

        This method is responsible for cleaning up the wrapped resource if it
        is a context manager.

        -   If `self._resource` is `None`, no action is performed.
        -   If `self._resource` has an `__exit__` method, it calls
            `self._resource.__exit__(exc_type, exc_val, exc_tb)`. The return
            value of this delegated `__exit__` method is propagated.
        -   Otherwise (if `self._resource` is not `None` but does not have
            an `__exit__` method), no action is performed.

        Parameters
        ----------
        exc_type : type or None
            The exception type, if an exception was raised inside the `with`
            block; `None` otherwise.

        exc_val : BaseException or None
            The exception value, if an exception was raised; `None` otherwise.

        exc_tb : traceback or None
            The traceback object, if an exception was raised; `None` otherwise.

        Returns
        -------
        bool or None
            `True` if the exception (if any) was handled, `False` if not.
            Returns `None` if no exception occurred or if the wrapped resource
            is not a context manager.
        """
        if self._resource is not None and hasattr(self._resource, "__exit__"):
            # If the resource has an __exit__ method, call it
            return self._resource.__exit__(exc_type, exc_val, exc_tb)
        # Otherwise, do nothing for None resources or non-context managers
        return None  # Explicitly return None if no exception was handled by the wrapped resource


def open_raster_or_none(
    apath: Optional[str], *args: Any, **kwargs: Any
) -> Optional[rasterio.io.DatasetReader]:
    """
    Opens a raster file with Rasterio, or returns None if the path is None.

    This utility function provides a convenient way to attempt opening a raster
    file. If `apath` is `None`, it directly returns `None` without attempting
    to open a file, which is useful for optional inputs. Otherwise, it
    delegates to `rasterio.open()`.

    Parameters
    ----------
    apath : str or None
        The path to the raster file to open.
        If `None`, the function returns `None`.

    *args
        Additional positional arguments to pass to `rasterio.open()`.

    **kwargs
        Additional keyword arguments to pass to `rasterio.open()`.

    Returns
    -------
    rasterio.io.DatasetReader or None
        A `rasterio.io.DatasetReader` object if `apath` is a valid path and
        the file is successfully opened. Returns `None` if `apath` is `None`.

    """
    if apath is None:
        return apath
    else:
        return rasterio.open(apath, *args, **kwargs)


def safe_raster_open(apath: Optional[str], *args: Any, **kwargs: Any) -> SafeContext:
    """
    Provides a safe context manager for opening raster files, or handling None
    paths.

    This function acts as a convenient alias, streamlining the pattern of
    using `SafeContext` with the `open_raster_or_none` helper. It creates
    a `SafeContext` instance, passing it the result of `open_raster_or_none`.
    This allows for clean `with` statements where the resource can be either
    an opened `rasterio` dataset or `None` if the path was `None`.

    Parameters
    ----------
    apath : str or None
        The path to the raster file to open. If `None`, the `SafeContext`
        will yield `None` when entered.

    *args
        Additional positional arguments to pass through to `rasterio.open()`
        via `open_raster_or_none`.

    **kwargs
        Additional keyword arguments to pass through to `rasterio.open()`
        via `open_raster_or_none`.

    Returns
    -------
    SafeContext
        An instance of `SafeContext` that will manage the safe opening and
        closing of the raster file (or `None` handling) within a `with`
        statement.

    Examples
    --------
    >>> # Create a dummy raster for the example
    >>> import os
    >>> with rasterio.open("example_raster.tif", 'w', driver='GTiff',
    ...                    height=1, width=1, count=1, dtype='uint8') as dst:
    ...     dst.write(np.array([[10]], dtype='uint8'), 1)
    >>>
    >>> # Using with a valid path
    >>> with safe_raster_open("example_raster.tif", 'r') as src:
    ...     if src:
    ...         print(f"Raster opened successfully: {src.name}")
    ...         # Perform operations with 'src'
    ...     else:
    ...         print("Raster was not opened.")
    >>> os.remove("example_raster.tif") # Clean up
    >>>
    >>> # Using with a None path
    >>> with safe_raster_open(None) as src_none:
    ...     if src_none:
    ...         print(f"Raster opened successfully: {src_none.name}")
    ...     else:
    ...         print("Path was None, raster not opened.")
    >>>
    >>> # Handling a non-existent file (Rasterio will raise an error)
    >>> try:
    ...     with safe_raster_open("non_existent_file.tif", 'r') as src_invalid:
    ...         pass
    ... except rasterio.errors.RasterioIOError as e:
    ...     print(f"Caught expected Rasterio error: {e}")
    """
    return SafeContext(open_raster_or_none(apath, *args, **kwargs))
