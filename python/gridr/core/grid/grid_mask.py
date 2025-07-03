# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Grid mask module
"""
from enum import IntEnum
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, NoReturn, Optional

import numpy as np
import shapely

from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.array_window import (window_shape, window_check,
        window_apply)
from gridr.core.grid.grid_utils import oversample_regular_grid
from gridr.core.grid.grid_rasterize import (GeometryType, grid_rasterize,
        GridRasterizeAlg, ShapelyPredicate)

class Validity(IntEnum):
    """Define the GridR convention regarding validity
    """
    INVALID = 0
    VALID = 1

def build_mask(
        shape: Tuple[int, int],
        resolution: Tuple[int, int],
        out: np.ndarray,
        geometry_origin: Tuple[float, float],
        geometry_pair: Tuple[Optional[GeometryType], Optional[GeometryType]],
        mask_in: Optional[np.ndarray],
        mask_in_target_win: np.ndarray,
        mask_in_resolution: Optional[Tuple[int, int]],
        oversampling_dtype: np.dtype,
        mask_in_binary_threshold: float = 0.999,
        rasterize_kwargs: Optional[Dict] = None,
        init_out: bool = False,
        ) -> Optional[np.ndarray]:
    """Create a binary mask associated with a grid.

    This method operates solely on raster data and does not perform I/O.
    It combines information from two distinct mask types to build a binary
    raster mask at a target resolution (currently only full resolution,
    i.e., (1,1), is implemented).

    1.  **Input Raster Mask**: Provided as `mask_in`, this is an
        optional binary raster mask, potentially at a lower resolution.
        It typically matches the grid's shape and resolution. If set,
        `mask_in_resolution` becomes mandatory.
        The `mask_in_target_win` argument can define a window for the
        resampled mask, specified in the output resolution's coordinate
        system. As resampling may yield float values,
        `mask_in_binary_threshold` binarizes the result: values
        greater than or equal to this threshold are `1` (valid),
        otherwise `0`.

    2.  **Vector Geometry Mask**: The `geometry_pair` argument defines
        this mask using vectors. It expects a tuple with two
        `GeometryType` elements:
        * The **first element** represents **valid** geometries.
        * The **second element** represents **invalid** geometries.
        (For details on `GeometryType`, see the `grid_rasterize` module.)

        Both geometry elements are processed sequentially to generate
        the rasterized vector mask:
        * **Valid Geometry Rasterization**: The first element is
            used in a `grid_rasterize` call, with `inner_value` as
            `Validity.VALID` and `outer_value` as `Validity.INVALID`.
            This marks the interior (and potentially contour) of the
            geometry as valid. If `None`, the resulting raster is
            entirely valid by convention.
        * **Invalid Geometry Rasterization**: The second element
            is then used in another `grid_rasterize` call. Here,
            `inner_value` is `Validity.INVALID` and `outer_value` is
            `Validity.VALID`. This produces a second raster where the
            interior (and contour) of this geometry is marked invalid.
        * **Final Mask Merge**: A unary "AND" operation merges the
            two rasters to yield the final geometry raster.

    A pixel is considered invalid if it is masked by the input raster, falls
    outside the **valid** geometry, or lies within the **invalid** geometry.

    The rasterization process aligns with the output coordinate system,
    defined by the `shape`, `resolution`, and `geometry_origin` arguments.
    You can pass a preallocated output buffer via the `out` argument;
    if provided, it must be consistent with the given `shape`.
    
    If neither an input mask (`mask_in`) nor a geometry pair (`geometry_pair`)
    is provided, the `out` buffer will not be modified unless the 'init_out' 
    argument is True. In such cases, it's the user's responsibility to ensure
    'out' is appropriately initialized, as its contents might otherwise be 
    non-conforming. If 'init_out' is True, the 'out' buffer will be filled with 
    'Validity.VALID'.

    **Conventions:**

    * **Invalid (masked) pixels** are `Validity.INVALID` (0);
        otherwise, they're `Validity.VALID` (1).
    * Geometry points use `(x, y)` coordinates, where `x` is the column
        and `y` is the row.
    * `shape`, `resolution`, and `geometry_origin` are provided as
        `(value for row, value for column)`. Note that `geometry_origin`'s
        convention differs from geometry point definitions.

    Args:
    -----
    
    shape : (Tuple[int, int])
        The shape of the output mask. If notgiven, it's defined from 
        `mask_in_target_win` or `out` in that priority order.
    
    resolution : (Tuple[int, int])
        The output mask's resolution.
        Only full resolution ((1,1)) is currently implemented. It's used for 
        resampling the input mask and rasterizing geometries.
    
    out : (np.ndarray)
        An optional preallocated buffer to store the result.
    
    geometry_origin : (Tuple[float, float])
        Geometric coordinates mapped to the output array's (0,0) pixel. This
        argument is mandatory if `geometry_pair` is set.
    
    geometry_pair : (Tuple[GeometryType, GeometryType])
        A tuple containing:
        - The first element (`GeometryType`): Represents **valid** geometries.
        - The second element (`GeometryType`): Represents **invalid** geometries.
    
    mask_in : (Optional[np.ndarray])
        Optional input raster mask.
    
    mask_in_target_win : (np.ndarray)
        An optional production window as
        `((first_row, last_row), (first_col, last_col))` for a 2D mask.
    
    mask_in_resolution : (Optional[Tuple[int, int]])
        Resolution in row and column for the input raster mask.
    
    oversampling_dtype : (np.dtype)
        The floating-point data type to use for mask oversampling.
    
    mask_in_binary_threshold : (float)
        For binary output, values greater than or equal to this threshold are 
        `1`, `0` otherwise.

    rasterize_kwargs : (Optional[Dict])
        Dictionary of parameters for the rasterization process, e.g.,
        `{'alg': GridRasterizeAlg.SHAPELY,
          'kwargs_alg': {'shapely_predicate': ShapelyPredicate.COVERS}}`.
    
    init_out : (bool)
        An option to force input 'out' buffer to be filled with Validity.VALID
        before any mask operation.
    
    Returns:
    --------
        Optional[np.ndarray]
            The created binary mask, or `None` if `out` was provided and the 
            operation was in-place.
    """
    ret = None
    # -- Perform some checks on arguments and init optional arguments
    if shape is None or resolution is None:
        raise ValueError("You must provide both the 'shape' and 'resolution' "
                "arguments")
    if ~np.all(resolution == (1,1)):
        raise ValueError("Output resolution different from full resolution have"
                " not been implemented yet")
    
    has_geometry = (geometry_pair is not None) and (geometry_pair != (None, None))
    if has_geometry and geometry_origin is None:
        raise ValueError("You must provide the 'geometry_origin' argument "
                "in order to use rasterization through the 'geometry' argument")
    
    if mask_in is not None and not mask_in_resolution:
        raise ValueError("You must provide the 'mask_in_resolution' argument")
        
    # Init output buffer if not given
    if out is None:
        out = np.full(shape, Validity.VALID, dtype=np.uint8)
        ret = out
    elif init_out:
        out[:] = Validity.VALID
    
    if ~np.all(out.shape == shape):
        raise ValueError("The values of the 2 arguments 'out' and 'shape' does "
                "not match.")
    
    if oversampling_dtype is not None:
        if not np.issubdtype(oversampling_dtype, np.floating):
            raise ValueError("The value of argument 'oversampling_dtype' is not"
                    " a floating type")
    else:
        raise ValueError("You must precise argument 'oversampling_dtype'")
    
    # At last check that the window lies in the full_resolution input mask
    # if given
    if mask_in is not None:
        if mask_in_target_win is None:
            # Define a correct 2d window matching the array dimensions
            mask_in_target_win = [(0, shape[i]-1) for i in range(2)]
        elif ~np.all(shape == window_shape(mask_in_target_win)):
            raise ValueError(f"The shapes of the 2 arguments 'shape' ({shape}) "
                    f"and 'mask_in_target_win' "
                    f"({window_shape(mask_in_target_win)}) does not match.")
                    
        mask_in_target_win = np.asarray(mask_in_target_win)
        
        # Compute the mask in profile at full resolution
        # FUTURE_WARNING : if resolution_out != 1 we will have to update this
        # code
        mask_in_full_res_profile = ArrayProfile(
                shape=((mask_in.shape[0]-1)*mask_in_resolution[0]+1,
                        (mask_in.shape[1]-1)*mask_in_resolution[1]+1),
                ndim=mask_in.ndim, dtype=mask_in.dtype)
        if not window_check(arr=mask_in_full_res_profile,
                win=mask_in_target_win, axes=None):
            raise ValueError(
                    "Target window error is not contained in input mask : "
                    f"\n\t Input mask : {mask_in_full_res_profile.shape}"
                    f"\n\t Window : {mask_in_target_win}")
    # -- End of argument's checks
    
    # Compute the full resolution binary mask if given
    merge = False
    if mask_in is not None:
        merge = True
        if mask_in_resolution[0] != 1 or mask_in_resolution[1] != 1:
            # We have to oversample the mask to the output resolution
            # FUTURE_WARNING : if resolution_out != (1,1) we will have to
            # reimplement this part
            _, out[:,:] = oversample_regular_grid(
                    grid = None,
                    grid_oversampling_row=mask_in_resolution[0],
                    grid_oversampling_col=mask_in_resolution[1],
                    grid_mask=mask_in,
                    grid_mask_binarize_precision = mask_in_binary_threshold,
                    grid_mask_dtype = out.dtype,
                    win = mask_in_target_win,
                    dtype = oversampling_dtype, # dtype used for interpolation
                    )
        else:
            # No oversampling - apply the window selection and binarize it
            # FUTURE_WARNING : if resolution_out != (1,1) we will have to
            # reimplement this part
            windows_mask_in = window_apply(arr=mask_in, win=mask_in_target_win)
            out[:,:] = np.abs(windows_mask_in) >= mask_in_binary_threshold
    
    # Rasterize the geometry if it has been set
    base_rasterize_args = {
        'grid_coords': None,
        'shape': shape,
        'origin': geometry_origin,
        'resolution': resolution,
        'win': None,
        'alg': rasterize_kwargs['alg'] if rasterize_kwargs else None,
        'reduce': False,
    }
    
    # Helper function to perform rasterization and merge
    def grid_rasterize_wrapper(
        geometry_to_rasterize: GeometryType,
        inner_val: int,
        outer_val: int,
        default_val: int,
        current_out_array: np.ndarray,
        merge_current_out_array: bool,
        temp_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Helper to rasterize a geometry and optionally merge it.
        Returns the result of the rasterization (temp or direct to out).
        """
        raster_args = base_rasterize_args.copy()
        raster_args['geometry'] = geometry_to_rasterize
        raster_args['inner_value'] = inner_val
        raster_args['outer_value'] = outer_val
        raster_args['default_value'] = default_val
        raster_args['dtype'] = None

        if merge_current_out_array:
            # If merging, rasterize to a new temp array
            raster_args['output'] = temp_array
            temp_raster_out = None
            if temp_array is None:
                raster_args['dtype'] = current_out_array.dtype
                # temp_raster_out is returned
                temp_raster_out = grid_rasterize(**raster_args)
            else:
                # here output buffer is filled, ie. temp_array
                _ = grid_rasterize(**raster_args)
                temp_raster_out = temp_array
            current_out_array[:] &= temp_raster_out[:] # Merge into current_out_array
            return current_out_array # Return the modified 'out' array
        else:
            # If not merging, can rasterize directly to 'out'
            raster_args['output'] = current_out_array
            return grid_rasterize(**raster_args) # Returns 'current_out_array'
    
    match geometry_pair:
        
        case None:
            pass
        
        case (None, None):
            pass
        
        case (geometry_valid, None) if geometry_valid is not None:
            # There is only valid geometry
            # Only valid geometry provided
            _ = grid_rasterize_wrapper( geometry_valid,
                    Validity.VALID, Validity.INVALID, Validity.VALID, out,
                    merge, None)
        
        case (None, geometry_invalid) if geometry_invalid is not None:
            
            # There is only invalid geometry
            _ = grid_rasterize_wrapper(geometry_invalid,
                    Validity.INVALID, Validity.VALID, Validity.VALID, out, 
                    merge, None)
        
        case (geometry_valid, geometry_invalid) \
                if geometry_valid is not None and geometry_invalid is not None:
             
            # In that case we will proceed in 2 passes :
            # - pass 1 : rasterize the valid geometry
            # - pass 2 : rasterize the invalid geometry
            
            # Here we are sure we will need another buffer - we allocate it here
            # in order to reuse it if needed for both pass 1 and pass 2.
            tmp_out = np.empty_like(out)
            
            # PASS 1 - on valid geometry
            _ = grid_rasterize_wrapper(geometry_valid,
                    Validity.VALID, Validity.INVALID, Validity.VALID, out,
                    merge, tmp_out)
            
            # PASS 2 - on invalid geometry
            _ = grid_rasterize_wrapper(geometry_invalid,
                    Validity.INVALID, Validity.VALID, Validity.VALID, out,
                    True, tmp_out)

    return ret