# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Grid rasterize module
"""
from enum import IntEnum
from typing import Tuple, Union, List, Dict, Optional

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import shapely

from gridr.core.utils.array_window import window_check, window_apply
from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.grid.grid_commons import (grid_regular_coords_1d,
        grid_regular_coords_2d,
        regular_grid_shape_origin_resolution,
        window_apply_grid_coords,
        window_apply_shape_origin_resolution,
        check_grid_coords_definition)

DFLT_GEOMETRY_BUFFER_DISTANCE = 1e-6

# Define a type alias for geometry input
GeometryType = Union[ shapely.geometry.Polygon, List[shapely.geometry.Polygon],
        shapely.geometry.MultiPolygon]


class GridRasterizeAlg(IntEnum):
    """Define the backend to use for rasterize.
    """
    RASTERIO_RASTERIZE = 1
    SHAPELY = 2


class ShapelyPredicate(IntEnum):
    """Define the shapely predicates used for rasterize
    COVERS: covers(a,b) returns True if b is within a or b lies in the contour
    CONTAINS: contains(a,b) returns True if b is within a. (It will returns
            False if b only touches a)
    INTERSECTS: intersects(a,b) : should produce same result as COVERS but less
            efficient.
    """
    COVERS = 1
    CONTAINS = 2
    INTERSECTS = 3


# Prepare geometries so that it correspond to a list of Polygons
def geometry_to_polygon_list(geom: shapely.geometry
        ) -> List[shapely.geometry.Polygon]:
    """Convert a geometry supposed to be a MultiPolygon or a Polygon to a
    list of polygons.
    
    Args:
        geom: the geometry to convert
    
    Returns:
        a list of polygons or an empty list if geometry's type mismatch.
    """
    geom_list = []
    if geom.geom_type == 'MultiPolygon':
        geom_list = [poly for poly in geom.geoms]
    elif geom.geom_type == 'Polygon':
        geom_list = [geom]
    return geom_list


def _grid_rasterize_check_params(
        grid_coords: Optional[Tuple[np.ndarray, np.ndarray]],
        shape: Optional[Tuple[int, int]],
        origin: Optional[Tuple[float, float]],
        resolution: Optional[Tuple[int, int]],
        win: Optional[np.ndarray],
        geometry: Optional[GeometryType],
        output: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
        reduce: bool = False,
        ) -> Union[np.ndarray, np.uint8]:
    """Check grid_rasterize method parameters.
    
    Args:
        grid_coords: grid corresponding coordinates. If not given they are
                computed with shape, origin and resolution arguments.
                The grid coordinates are given as tuple of 1d or 2d arrays 
                containing the columns and the rows of pixels centroïds given
                in the same frame as the geometry.
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
        win: the production window given as a list of tuple containing the
                first and last index for each dimension. The window is defined
                in regards to the given coordinates.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
        geometry: geometry to rasterize on the grid. This can be either a
                simple Polygon, a MultiPolygon or a list of Polygons.
        output: if not None, the result will be put in output. Not working if
                reduce is set to true. If a window is defined the output must be
                the exact same size.
        dtype: output dtype
        reduce: a boolean option that returns the corresponding scalar (0 or 1)
                if the resulting raster if fully filled with that scalar
        
    Returns:
        A tuple containing updated variables : grid_coords, dtype, shape_out,
        win and polygons
    """
    # Check the grid coords definition
    grid_coords = check_grid_coords_definition(grid_coords, shape, origin,
            resolution)
    
    # Check that both reduce and output are not set to True
    if reduce and output is not None:
        raise ValueError("The arguments 'reduce' and 'output' cannot be set "
                "at the same time")
    
    # Check that both output and dtype are not set to True
    if dtype is not None and output is not None:
        raise ValueError("The arguments 'dtype' and 'output' cannot be set "
                "at the same time")
    elif output is not None:
        dtype = output.dtype
    
    # Compute output shape before windowing
    # If grid_coords is passed as argument then the output shape is the shape of
    # the grid coords array.
    # If shape is passed as argument then the output shape is set with its value
    shape_out = None
    if grid_coords:
        # Shape out here is 2d
        shape_out, _, _ = regular_grid_shape_origin_resolution(grid_coords)
    else:
        shape_out = shape
    # Create an array profile for output in order to use window utils
    array_profile_out = ArrayProfile(shape=shape_out, ndim=2, dtype=dtype)
    
    # Set window to full array if not given
    if win is None:
        # Define a correct 2d window matching the array dimensions
        win = [(0, shape_out[0]-1), (0, shape_out[1]-1)]
    else:
        # Check the window is ok
        if not window_check(array_profile_out, win):
            raise Exception("The given 'window' is outside the grid domain of "
                    "definition.")
        # Update the output shape
        shape_out = win[:,1] - win[:,0] + 1
    win = np.asarray(win)
    
    # Construct list of polygons
    polygons = []
    try:
        polygons = geometry_to_polygon_list(geometry)
    except AttributeError:
        # We suppose here geometries are passed as a list of geometries
        for geom in geometry:
            polygons.extend(geometry_to_polygon_list(geom))
    
    return grid_coords, dtype, shape_out, win, polygons


def grid_rasterize(
        grid_coords: Optional[Tuple[np.ndarray, np.ndarray]],
        shape: Optional[Tuple[int, int]],
        origin: Optional[Tuple[float, float]],
        resolution: Optional[Tuple[int, int]],
        win: Optional[np.ndarray],
        inner_value: int,
        outer_value: int,
        default_value: int,
        geometry: GeometryType,
        geometry_buffer_dst: Optional[float] = DFLT_GEOMETRY_BUFFER_DISTANCE,
        alg: GridRasterizeAlg = GridRasterizeAlg.RASTERIO_RASTERIZE,
        output: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
        reduce: bool = False,
        **kwargs_alg,
        ) -> Union[np.ndarray, np.uint8]:
    """
    Generates a raster mask based on the spatial relationship between grid cell
    centroids and the input geometry.

    Each pixel in the output raster will be set to `inner_value` if its 
    corresponding grid cell centroid is considered to be within the geometry, 
    according to the optionally specified predicate.
    Otherwise, the mask pixel will be set to `outer_value`.

    If the input geometry is empty (e.g., no polygons are defined), the entire
    mask will be populated with `default_value`.
    
    Args:
        grid_coords: grid corresponding coordinates. If not given they are
                computed with shape, origin and resolution arguments.
                The grid coordinates are given as tuple of 1d or 2d arrays 
                containing the columns and the rows of pixels centroïds given
                in the same frame as the geometry.
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
        win: the production window given as a list of tuple containing the
                first and last index for each dimension. The window is defined
                in regards to the given coordinates.
                e.g. for a dimension 2 : 
                ((first_row, last_row), (first_col, last_col))
        inner_value: The value to use for the interior of the union of polygons.
        outer_value: The value to use for the exterior of the union of polygons.
        default_value: The value to use to fill the output if no polygons is
                given.
        geometry: geometry to rasterize on the grid. This can be either a
                simple Polygon, a MultiPolygon or a list of Polygons.
        geometry_buffer_dst: an optional distance to apply to dilate (positive)
                or erode (negative) the geometries. These may be needed for the
                RASTERIO_RASTERIZE backend in order to ensure that polygons
                edge's corners will be burned.
        alg: backend to use for rasterization. Some backend may need additionnal
                arguments given by the kwargs_alg keyword arguments dictionnary.
        output: if not None, the result will be put in output. Not working if
                reduce is set to true. If a window is defined the output must be
                the exact same size.
        dtype: output dtype. Please note that 'bool' type is not available with
                the GridRasterizeAlg.RASTERIO_RASTERIZE algorithm.
        reduce: a boolean option that returns the corresponding scalar (0 or 1)
                if the resulting raster if fully filled with that scalar
        kwargs_alg: additionnal dictionnary of arguments needed for rasterize
                backend.
    
    Returns:
        The binary raster mask or scalar if reduce is True and the mask only
        contains a unique value.
    """
    raster = None
    
    # check parameters and update them
    grid_coords, dtype, shape_out, win, polygons = \
            _grid_rasterize_check_params(grid_coords, shape, origin, resolution,
            win, geometry, output, dtype, reduce)
        
    # Test we got polygons to rasterize
    if len(polygons) > 0:
        
        if geometry_buffer_dst is not None:
            # Dilate or erode the polygons through shapely.buffer method
            polygons = [poly.buffer(distance=geometry_buffer_dst, quad_segs=1,
                    cap_style=1, join_style=1, single_sided=False)
                    for poly in polygons]
    
        if alg == GridRasterizeAlg.RASTERIO_RASTERIZE:
            # TODO : to be tested !
            if grid_coords is not None:
                shape, origin, resolution \
                        = regular_grid_shape_origin_resolution(grid_coords)
            # If window is not on full data
            if ~np.all(win[:,1]+1 == shape):
                shape, origin, resolution \
                        = window_apply_shape_origin_resolution(shape, origin,
                        resolution, win)
            
            kwgs = {}
            if output is None:
                kwgs["dtype"] = dtype
            raster = rasterize_polygons_rasterio_rasterize(shape, origin,
                    resolution, polygons, inner_value, outer_value, output,
                    **kwgs)
        
        elif alg == GridRasterizeAlg.SHAPELY:
            kwgs = {}
            for kwarg_key in ("shapely_predicate", ):
                try:
                    kwgs[kwarg_key] = kwargs_alg[kwarg_key]
                except KeyError:
                    pass
            
            # Compute grid coordinates if not given
            if not grid_coords:
                grid_coords = grid_regular_coords_2d(shape, origin, resolution,
                        sparse=False)
            
            # Apply window - check has already been performed previously
            grid_coords = window_apply_grid_coords(grid_coords, win,
                    check=False)
           
            if output is None:
                kwgs["dtype"] = dtype
            # Call the rasterize method
            raster = rasterize_polygons_shapely(polygons=polygons, 
                    inner_value=inner_value, outer_value=outer_value,
                    grid_coords=grid_coords, output=output, **kwgs)
        
        else:
            raise ValueError(f"Unknown 'alg' {kwargs_alg}")
        
        if reduce:
            if np.all(raster==inner_value):
                raster = inner_value # pylint: disable=R0204
            elif np.all(raster!=inner_value):
                raster = outer_value # pylint: disable=R0204
    else:
        if reduce:
            raster = default_value # pylint: disable=R0204
        else:
            if output:
                output[:,:] = default_value
            else:
                raster = np.empty(shape_out, dtype=dtype)
                raster[:, :] = default_value
        
    return raster


def rasterize_polygons_shapely(
        polygons: List[shapely.geometry.Polygon],
        inner_value: int,
        outer_value: int,
        grid_coords:Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        shape: Optional[Tuple[float, float]] = None,
        origin: Optional[Tuple[float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        shapely_predicate: ShapelyPredicate = ShapelyPredicate.COVERS,
        output: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
        ) -> np.ndarray:
    """Rasterize list of polygons on a grid as a binary raster using shapely.
    
    The raster's pixels will contain `inner_value` if the corresponding grid's 
    pixel centroïd is considered in the geometry with regards to the optional 
    chosen predicate.
    Otherwise the raster's pixels will contain `outer_value`.
    
    Args:
        polygons: geometry to rasterize on the grid given as a list of polygons.
        inner_value: The value to use for the interior of the union of polygons.
        outer_value: The value to use for the exterior of the union of polygons.
        grid_coords: coordinates of pixel centers given as a 3d array or a tuple
                of 2 2d arrays/
                First axis contains index for columns and rows coordinates.
        shape: grid output shape as a tuple of integer (number of rows, number 
                of columns).
        origin: grid origin as a tuple of float (origin's row coordinate, 
                origin's column coordinate)
        resolution: grid resolution as a tuple of integer (row resolution,
                columns resolution)
        output: if not None, the result will be put in output. Not working if
                reduce is set to true. If a window is defined the output must be
                the exact same size.
                Please note output may be initialized with values. The method
                resets them.
        dtype: output dtype ; only used if output is not given
        shapely_predicate: the predicate to use for mask computation.
    
    Returns:
        The binary raster mask
    """
    # Not yet implemented : raise exception if output is given
    if (output is not None and dtype is not None) \
            or (output is None and dtype is None):
        raise ValueError("You should either provide the an output buffer or the"
                " dtype argument")
    
    if grid_coords is not None \
            and shape is None and origin is None and resolution is None:
        pass
    elif grid_coords is None \
            and shape is not None and origin is not None \
            and resolution is not None:
        grid_coords = grid_regular_coords_2d(shape, origin, resolution,
                sparse=False)
    else:
        raise ValueError("You should either provide the grid_coords argument "
                "or the 3 arguments 'shape', 'origin' and 'resolution'")
    
    xx, yy = grid_coords
    if xx.ndim == 1 and yy.ndim == 1:
        xx, yy = np.meshgrid(xx, yy, indexing='xy', sparse=False)
    
    # Check output shape
    if output is not None and ~np.all(output.shape == xx.shape):
        raise ValueError("The output buffer's shape does not match the grid's "
                "shape")
    
    points = [shapely.Point(x,y) for x,y in zip(xx.flatten(), yy.flatten())]
    
    # Prepare geometries in order to optimize computation
    # This methods affects objects inplace
    _ = [shapely.prepare(polygon) for polygon in polygons]
    
    # TODO : STRrtree
    if inner_value not in [0, 1]:
        raise ValueError("The argument 'inner_value' must have a binary value "
                "(0 or 1)")
    
    if outer_value not in [0, 1]:
        raise ValueError("The argument 'inner_value' must have a binary value "
                "(0 or 1)")
    
    if inner_value == outer_value:
        raise ValueError("The argument 'inner_value' and 'outer_value' must be "
                "different")
    
    # Here we adopt the shapely convention :
    # - 0 is used for the exterior
    # - 1 is used for the interior
    # A final inversion is performed in case inner_value is 0
    
    # Init mask
    if output is not None:
        # reset output with zeros (False)
        output[:,:] = 0
        # Reshape to flatten => this will not perform copy for contiguous arrays
        mask = output.reshape(-1)
    else:
        mask = np.zeros(len(points), dtype=bool, order='C')
    
    if shapely_predicate == ShapelyPredicate.COVERS:
        for polygon in polygons:
            mask |= shapely.covers(polygon, points)
            
    elif shapely_predicate == ShapelyPredicate.CONTAINS:
        for polygon in polygons:
            mask |= shapely.contains(polygon, points)
            
    elif shapely_predicate == ShapelyPredicate.INTERSECTS:
        for polygon in polygons:
            mask |= shapely.intersects(polygon, points)
        
    if output is not None:
        # Invert the mask if inner_value is 1
        if output.dtype == bool:
            if inner_value == 0:
                np.invert(output, out=output)
            else:
                # convert in bool
                output[:,:] = output.astype(bool)
        else:
            if inner_value == 0:
                output[:,:] = np.where(output,0,1)
            # No return
        mask = None
    else:
        # Invert the mask if inner_value is 1
        if inner_value == 0:
            mask = (~mask).reshape(xx.shape).astype(dtype)
        else:
            mask = mask.reshape(xx.shape).astype(dtype)
    
    return mask


def rasterize_polygons_rasterio_rasterize(
        shape: Tuple[float, float],
        origin: Tuple[float, float],
        resolution: Tuple[float, float],
        polygons: List[shapely.geometry.Polygon],
        inner_value: int,
        outer_value: int,
        output: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,
        ) -> Union[np.ndarray, int]:
    """Rasterize a geometry on a grid
    
    That method used rasterio.features.rasterize in order to create the raster.
    This method implies the definition of an AffineTransform. It is defined
    using the origin and resolution :
    
        | a b c |   | res[1]     0.     O[1] |
    A = | d e f | = |   0.     res[0]   O[0] |
        | g h i |   |   0.       0.      1.  |
    
    Args:
        shape: target raster size given as a tuple (number of rows, number of
                columns).
        origin: the coordinates (row, col) of the raster first element (0,0)
                in the image coordinates reference system used by the geometry.
        resolution: the grid relative pixel size towards the geometry 
                coordinate reference system.
        polygons: list of polygons.
                The polygons coordinates must be defined in the same reference
                frame as that intrinsic to the image targeted by the raster, 
                considering the scale factor (resolution) and with a possible
                shift of its origin.
                The coordinates of the polygons are here supposed to be given
                following the standard (x: col, y: row) order.
        inner_value: The value to use for the interior of the union of polygons.
        outer_value: The value to use for the exterior of the union of polygons.
        output: if not None, the result will be put in output. Not working if
                reduce is set to true. If a window is defined the output must be
                the exact same size.
        dtype: output dtype ; only used if output is not given. Please note that
                'bool' type is not available.
        
    Returns:
        the binary mask
    """
    if (output is not None and dtype is not None) \
            or (output is None and dtype is None):
        raise ValueError("You should either provide the output buffer or the"
                " dtype argument")
    # Check output shape
    if output is not None and shape is not None \
            and ~np.all(output.shape == shape):
        raise ValueError("The output buffer's shape does not match the shape "
                "argument")
        
    # RasterIO method is parametrized through an AffineTransform (a, b, c, d, e,
    # f, g, h, i) corresponding to the affine transform matrix :
    #
    # | a b c |
    # | d e f |
    # | g h i |
    #
    # Usually g, h, i are respectively set to 0, 0, 1
    transform = Affine(resolution[1], 0, origin[1]-0.5*resolution[1],
              0, resolution[0], origin[0]-0.5*resolution[0],
              0, 0, 1)
    
    # We could also use the from_origin method :
    #  transform = rasterio.transform.from_origin(
    #        west=origin[1]-0.5*resolution[1], 
    #        north=origin[0]-0.5*resolution[0], 
    #        xsize=resolution[1], 
    #        ysize=-resolution[0]
    #        )
    
    kwargs = {
            'shapes': polygons,
            'transform': transform,
            'all_touched': False,
            'default_value': inner_value,
            'fill': outer_value}
    if output is not None:
        # reset output with fill value
        output[:,:] = outer_value
        kwargs['out'] = output
    else:
        kwargs['out_shape'] = (shape[0], shape[1])
        kwargs['dtype'] = dtype
    raster = rasterize(**kwargs)
    return raster
    