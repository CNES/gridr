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
Grid rasterize module
"""
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine

from gridr.core.grid.grid_commons import (
    check_grid_coords_definition,
    grid_regular_coords_2d,
    regular_grid_shape_origin_resolution,
    window_apply_grid_coords,
    window_apply_shape_origin_resolution,
)
from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.array_window import window_check

DFLT_GEOMETRY_BUFFER_DISTANCE = 1e-6

# Define a type alias for geometry input
GeometryType = Union[
    shapely.geometry.Polygon, List[shapely.geometry.Polygon], shapely.geometry.MultiPolygon
]
"""
Type alias for valid geometry inputs.

This type alias simplifies type hints for functions that accept various
Shapely geometry types or collections of them. It covers single polygons, lists
of polygons, and multi-polygons, which are common inputs for rasterization
operations.

Examples
--------
>>> from shapely.geometry import Polygon, MultiPolygon
>>> poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
>>> poly_list = [poly1, Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])]
>>> multi_poly = MultiPolygon(poly_list)

>>> def process_geometry(geom: GeometryType):
...     # Function body
...     pass

>>> process_geometry(poly1)
>>> process_geometry(poly_list)
>>> process_geometry(multi_poly)
"""


class GridRasterizeAlg(IntEnum):
    """Define the backend algorithm to use for rasterization.

    This enumeration specifies the different libraries or approaches that
    can be employed to perform the rasterization of geometric features
    onto a grid.

    Attributes
    ----------
    RASTERIO_RASTERIZE : int
        Uses `rasterio.features.rasterize` as the backend for rasterization.
        This typically offers high performance and robustness for geospatial
        rasterization tasks.

    SHAPELY : int
        Uses Shapely's geometric operations for rasterization. This might
        involve iterating over geometries and using Shapely's predicates
        (e.g., `contains`, `intersects`) to determine cell inclusion.
        It can be useful for fine-grained control over predicate logic but might
        be less performant for large-scale rasterization compared to
        `rasterio.features.rasterize`.
    """

    RASTERIO_RASTERIZE = 1
    SHAPELY = 2


class ShapelyPredicate(IntEnum):
    """Define the shapely predicates used for rasterization.

    These predicates determine how geometric features (typically polygons)
    are evaluated against raster cells during the rasterization process.

    Attributes
    ----------
    COVERS : int
        `covers(a, b)` returns `True` if geometry `b` is entirely within
        geometry `a` or if `b` lies on the contour of `a`.

    CONTAINS : int
        `contains(a, b)` returns `True` if geometry `b` is strictly within
        geometry `a`. It returns `False` if `b` only touches `a` (i.e.,
        intersects only at the boundary).

    INTERSECTS : int
        `intersects(a, b)` returns `True` if the interior or boundary of
        geometry `a` intersects the interior or boundary of geometry `b`.
        This predicate should produce the same geometric results as `COVERS`
        in many rasterization contexts but is generally less efficient for
        this specific use case.
    """

    COVERS = 1
    CONTAINS = 2
    INTERSECTS = 3


# Prepare geometries so that it correspond to a list of Polygons
def geometry_to_polygon_list(geom: shapely.geometry) -> List[shapely.geometry.Polygon]:
    """Convert a Shapely Polygon or MultiPolygon geometry to a list of polygons.

    This function takes a single `shapely.geometry.Polygon` or
    `shapely.geometry.MultiPolygon` and converts it into a uniform list of
    individual `shapely.geometry.Polygon` objects. This is useful for
    processing workflows that require iterating over individual polygons.

    Parameters
    ----------
    geom : Union[shapely.geometry.Polygon, shapely.geometry.MultiPolygon]
        The geometry to convert. Expected types are `Polygon` or `MultiPolygon`.

    Returns
    -------
    List[shapely.geometry.Polygon]
        A list of `shapely.geometry.Polygon` objects extracted from the input
        geometry. If the input `geom` is a `Polygon`, the list will contain just
        that polygon. If it's a `MultiPolygon`, the list will contain all its
        constituent polygons. Returns an empty list if the

    """
    geom_list = []
    if geom.geom_type == "MultiPolygon":
        geom_list = list(geom.geoms)
    elif geom.geom_type == "Polygon":
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
) -> Tuple[
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    Optional[np.dtype],
    Tuple[int, int],
    np.ndarray,
    List[shapely.geometry.Polygon],
]:
    """Check and preprocess parameters for the grid_rasterize method.

    This internal helper function validates and prepares all input parameters
    required by the `grid_rasterize` method. It ensures consistency between
    grid definition arguments, handles default windowing, and converts input
    geometries into a standardized list of polygons.

    Parameters
    ----------
    grid_coords : Optional[Tuple[np.ndarray, np.ndarray]]
        Grid corresponding coordinates. If `None`, they are computed using
        `shape`, `origin`, and `resolution` arguments. The grid coordinates
        are typically given as a tuple of 1D or 2D arrays containing the
        columns and rows of pixel centroids, expressed in the same frame as
        the geometry.

    shape : Optional[Tuple[int, int]]
        Grid output shape as a tuple of integers (number of rows, number of
        columns). This is used if `grid_coords` is not provided.

    origin : Optional[Tuple[float, float]]
        Grid origin as a tuple of floats (origin's row coordinate, origin's
        column coordinate). Used if `grid_coords` is not provided.

    resolution : Optional[Tuple[int, int]]
        Grid resolution as a tuple of integers (row resolution, columns
        resolution). Used if `grid_coords` is not provided.

    win : Optional[np.ndarray]
        The production window given as a list of tuples containing the first
        and last index for each dimension. The window is defined in regards
        to the given coordinates. For example, for a 2D dimension:
        ``((first_row, last_row), (first_col, last_col))``.

    geometry : Optional[GeometryType]
        Geometry to rasterize on the grid. This can be either a single
        `shapely.geometry.Polygon`, a `shapely.geometry.MultiPolygon`, or a list
        of `shapely.geometry.Polygon` objects.

    output : Optional[np.ndarray], default None
        If not `None`, the rasterization result will be stored directly into
        this preallocated array. This option is not compatible if `reduce` is
        set to `True`. If a `win` is defined, the `output` array must have the
        exact same size as the windowed region.

    dtype : Optional[np.dtype], default None
        Desired data type for the output rasterized array. If `None`, and
        `output` is provided, the `dtype` of `output` is used. This parameter
        is mutually exclusive with `output`.

    reduce : bool, default False
        A boolean option. If `True`, and if the resulting raster is fully filled
        with a single scalar value (0 or 1), then that scalar value is returned
        instead of the full raster array. This option is mutually exclusive with
        `output`.

    Returns
    -------
    Tuple[
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        Optional[np.dtype],
        Tuple[int, int],
        np.ndarray,
        List[shapely.geometry.Polygon]
    ]
        A tuple containing the validated and updated parameters:

        -   **grid_coords**: Updated grid coordinates (NumPy array or tuple
            of arrays).
        -   **dtype**: Resolved output data type.
        -   **shape_out**: The computed output shape of the rasterized grid
            (tuple of integers).
        -   **win**: The determined window to apply, as a NumPy array.
        -   **polygons**: A standardized list of `shapely.geometry.Polygon`
            objects derived from the input `geometry`.

    Raises
    ------
    ValueError
        If both `reduce` and `output` arguments are set to `True`.

    ValueError
        If both `dtype` and `output` arguments are set simultaneously
        (as `output.dtype` should define it).

    Exception
        If the given `win` (window) is outside the domain of the grid definition.

    """
    # Check the grid coords definition
    grid_coords = check_grid_coords_definition(grid_coords, shape, origin, resolution)

    # Check that both reduce and output are not set to True
    if reduce and output is not None:
        raise ValueError("The arguments 'reduce' and 'output' cannot be set " "at the same time")

    # Check that both output and dtype are not set to True
    if dtype is not None and output is not None:
        raise ValueError("The arguments 'dtype' and 'output' cannot be set " "at the same time")
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
        win = [(0, shape_out[0] - 1), (0, shape_out[1] - 1)]
    else:
        # Check the window is ok
        if not window_check(array_profile_out, win):
            raise Exception("The given 'window' is outside the grid domain of " "definition.")
        # Update the output shape
        shape_out = win[:, 1] - win[:, 0] + 1
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
) -> Union[np.ndarray, np.uint8, None]:
    """Generates a raster mask based on the spatial relationship between
    grid cell centroids and the input geometry.

    Each pixel in the output raster will be set to `inner_value` if its
    corresponding grid cell centroid is considered to be within the geometry,
    according to the optionally specified predicate (for Shapely backend).
    Otherwise, the mask pixel will be set to `outer_value`.

    If the input geometry is empty (e.g., no polygons are defined), the entire
    mask will be populated with `default_value`.

    Parameters
    ----------
    grid_coords : Optional[Tuple[np.ndarray, np.ndarray]]
        Grid corresponding coordinates. If `None`, they are computed with
        `shape`, `origin`, and `resolution` arguments. The grid coordinates are
        given as a tuple of 1D or 2D arrays containing the columns and rows of
        pixel centroids, expressed in the same frame as the geometry.

    shape : Optional[Tuple[int, int]]
        Grid output shape as a tuple of integers (number of rows, number of
        columns).

    origin : Optional[Tuple[float, float]]
        Grid origin as a tuple of floats (origin's row coordinate, origin's
        column coordinate).

    resolution : Optional[Tuple[int, int]]
        Grid resolution as a tuple of integers (row resolution, columns
        resolution).

    win : Optional[np.ndarray]
        The production window given as a list of tuples containing the first and
        last index for each dimension. The window is defined with respect to the
        given coordinates. For example, for a 2D dimension:
        ``((first_row, last_row), (first_col, last_col))``.

    inner_value : int
        The value to use for the interior of the union of polygons (pixels
        considered inside the geometry).

    outer_value : int
        The value to use for the exterior of the union of polygons (pixels
        considered outside the geometry).

    default_value : int
        The value to use to fill the entire output raster if no valid polygons
        are provided in `geometry`.

    geometry : GeometryType
        Geometry to rasterize on the grid. This can be either a single
        `shapely.geometry.Polygon`, a `shapely.geometry.MultiPolygon`, or a list
        of `shapely.geometry.Polygon` objects.

    geometry_buffer_dst : Optional[float], default DFLT_GEOMETRY_BUFFER_DISTANCE
        An optional distance to apply to dilate (positive value) or erode
        (negative value) the geometries using `shapely.buffer`. This may be
        needed for the `RASTERIO_RASTERIZE` backend to ensure that polygon
        edge's corners are properly burned into the raster.

    alg : GridRasterizeAlg, default GridRasterizeAlg.RASTERIO_RASTERIZE
        The backend algorithm to use for rasterization. Some backends may
        require additional arguments, which can be passed via `kwargs_alg`.

    output : Optional[np.ndarray], default None
        If not `None`, the rasterization result will be stored directly into
        this preallocated array. This option is mutually exclusive with
        `reduce=True`. If a `win` is defined, the `output` array must be exactly
        the same size as the windowed region.

    dtype : Optional[np.dtype], default None
        Desired data type for the output rasterized array. If `None`, and
        `output` is provided, the `dtype` of `output` is used. Note that `bool`
        type is not available with the `RASTERIO_RASTERIZE` algorithm.

    reduce : bool, default False
        If `True`, and if the resulting raster is entirely filled with either
        `inner_value` or `outer_value`, then that corresponding scalar value (0
        or 1) is returned instead of the full raster array.
        This option is mutually exclusive with `output`.

    kwargs_alg : dict
        Additional dictionary of arguments needed for the chosen rasterize
        backend. For `GridRasterizeAlg.SHAPELY`, this might include
        `shapely_predicate` (e.g., `ShapelyPredicate.CONTAINS`).

    Returns
    -------
    Union[np.ndarray, np.uint8, None]
        The binary raster mask as a NumPy array, or a scalar integer
        (`inner_value` or `outer_value`) if `reduce` is `True` and the
        mask contains only a single unique value. Returns `None` if
        `output` was provided and the result was written in-place.

    Raises
    ------
    ValueError
        If both `reduce` and `output` arguments are set to `True`.

    ValueError
        If both `dtype` and `output` arguments are set simultaneously (as
        `output.dtype` should define it).

    ValueError
        If an unknown `alg` (backend) is specified.

    Exception
        If the given `win` (window) is outside the domain of the grid definition.

    """
    raster = None

    # check parameters and update them
    grid_coords, dtype, shape_out, win, polygons = _grid_rasterize_check_params(
        grid_coords, shape, origin, resolution, win, geometry, output, dtype, reduce
    )

    # Test we got polygons to rasterize
    if len(polygons) > 0:

        if geometry_buffer_dst is not None:
            # Dilate or erode the polygons through shapely.buffer method
            polygons = [
                poly.buffer(
                    distance=geometry_buffer_dst,
                    quad_segs=1,
                    cap_style=1,
                    join_style=1,
                    single_sided=False,
                )
                for poly in polygons
            ]

        if alg == GridRasterizeAlg.RASTERIO_RASTERIZE:
            # TODO : to be tested !
            if grid_coords is not None:
                shape, origin, resolution = regular_grid_shape_origin_resolution(grid_coords)
            # If window is not on full data
            if ~np.all(win[:, 1] + 1 == shape):
                shape, origin, resolution = window_apply_shape_origin_resolution(
                    shape, origin, resolution, win
                )

            kwgs = {}
            if output is None:
                kwgs["dtype"] = dtype
            raster = rasterize_polygons_rasterio_rasterize(
                shape, origin, resolution, polygons, inner_value, outer_value, output, **kwgs
            )

        elif alg == GridRasterizeAlg.SHAPELY:
            kwgs = {}
            for kwarg_key in ("shapely_predicate",):
                try:
                    kwgs[kwarg_key] = kwargs_alg[kwarg_key]
                except KeyError:
                    pass

            # Compute grid coordinates if not given
            if not grid_coords:
                grid_coords = grid_regular_coords_2d(shape, origin, resolution, sparse=False)

            # Apply window - check has already been performed previously
            grid_coords = window_apply_grid_coords(grid_coords, win, check=False)

            if output is None:
                kwgs["dtype"] = dtype
            # Call the rasterize method
            raster = rasterize_polygons_shapely(
                polygons=polygons,
                inner_value=inner_value,
                outer_value=outer_value,
                grid_coords=grid_coords,
                output=output,
                **kwgs,
            )

        else:
            raise ValueError(f"Unknown 'alg' {kwargs_alg}")

        if reduce:
            if np.all(raster == inner_value):
                raster = inner_value  # pylint: disable=R0204
            elif np.all(raster != inner_value):
                raster = outer_value  # pylint: disable=R0204
    else:
        if reduce:
            raster = default_value  # pylint: disable=R0204
        else:
            if output:
                output[:, :] = default_value
            else:
                raster = np.empty(shape_out, dtype=dtype)
                raster[:, :] = default_value

    return raster


def rasterize_polygons_shapely(
    polygons: List[shapely.geometry.Polygon],
    inner_value: int,
    outer_value: int,
    grid_coords: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    shape: Optional[Tuple[float, float]] = None,
    origin: Optional[Tuple[float, float]] = None,
    resolution: Optional[Tuple[float, float]] = None,
    shapely_predicate: ShapelyPredicate = ShapelyPredicate.COVERS,
    output: Optional[np.ndarray] = None,
    dtype: Optional[np.dtype] = None,
) -> Optional[np.ndarray]:
    """Rasterizes a list of polygons onto a grid, producing a binary raster
    using Shapely's geometric predicates.

    Each pixel in the output raster will be set to `inner_value` if its
    corresponding grid cell centroid is considered to be within the geometry
    (union of polygons), according to the chosen `shapely_predicate`.
    Otherwise, the pixel will be set to `outer_value`.

    Parameters
    ----------
    polygons : List[shapely.geometry.Polygon]
        A list of `shapely.geometry.Polygon` objects to rasterize on the grid.

    inner_value : int
        The value to use for pixels whose grid cell centroids fall within the
        interior (or boundary, depending on predicate) of the union of polygons.
        Must be either 0 or 1.

    outer_value : int
        The value to use for pixels whose grid cell centroids fall outside the
        exterior of the union of polygons. Must be either 0 or 1.
        Must be different from `inner_value`.

    grid_coords : Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]], default None
        Coordinates of pixel centers. Can be:

            -   A 2D NumPy array with shape (N, 2) where N is the total number
                of points, and columns represent x and y coordinates.
            -   A tuple of two 2D NumPy arrays `(xx, yy)` representing the X and
                Y coordinates for each grid point (e.g., from `np.meshgrid`).
            -   A tuple of two 1D NumPy arrays `(x_coords, y_coords)` which will
                be expanded into 2D meshgrids.

        If provided, `shape`, `origin`, and `resolution` must be `None`.

    shape : Optional[Tuple[int, int]], default None
        Grid output shape as a tuple of integers `(num_rows, num_columns)`.
        If provided, `grid_coords` must be `None`, and `origin` and `resolution`
        must also be provided.

    origin : Optional[Tuple[float, float]], default None
        Grid origin as a tuple of floats `(origin_row_coordinate, origin_column_coordinate)`.
        Used with `shape` and `resolution` to compute `grid_coords` if
        `grid_coords` is `None`.

    resolution : Optional[Tuple[float, float]], default None
        Grid resolution as a tuple of floats `(row_resolution, column_resolution)`.
        Used with `shape` and `origin` to compute `grid_coords` if `grid_coords`
        is `None`.

    shapely_predicate : ShapelyPredicate, default ShapelyPredicate.COVERS
        The Shapely predicate to use for mask computation. Options are:
        `ShapelyPredicate.COVERS`, `ShapelyPredicate.CONTAINS`,
        or `ShapelyPredicate.INTERSECTS`.

    output : Optional[np.ndarray], default None
        An optional preallocated NumPy array buffer to store the result.
        If provided, the rasterization is performed in-place into this array.
        Its shape must match the target grid shape derived from `grid_coords`
        or `shape`/`origin`/`resolution`. If `output` is given, its `dtype` will
        be used, and the `dtype` argument should be `None`. The method will
        reset the values of this buffer before populating it.

    dtype : Optional[np.dtype], default None
        Desired NumPy data type for the output raster. This argument is only
        used if `output` is `None` (i.e., a new array needs to be allocated).

    Returns
    -------
    Optional[np.ndarray]
        The binary raster mask as a NumPy array. If `output` was provided,
        the rasterization is performed in-place, and this function returns
        `None`.

    Raises
    ------
    ValueError
        If neither `output` nor `dtype` is provided, or if both are provided.

    ValueError
        If `grid_coords` and `shape`/`origin`/`resolution` are inconsistently
        provided.

    ValueError
        If the `output` buffer's shape does not match the grid's shape.

    ValueError
        If `inner_value` or `outer_value` are not 0 or 1.

    ValueError
        If `inner_value` and `outer_value` are the same.

    """
    # Not yet implemented : raise exception if output is given
    if (output is not None and dtype is not None) or (output is None and dtype is None):
        raise ValueError("You should either provide the an output buffer or the" " dtype argument")

    if grid_coords is not None and shape is None and origin is None and resolution is None:
        pass
    elif (
        grid_coords is None and shape is not None and origin is not None and resolution is not None
    ):
        grid_coords = grid_regular_coords_2d(shape, origin, resolution, sparse=False)
    else:
        raise ValueError(
            "You should either provide the grid_coords argument "
            "or the 3 arguments 'shape', 'origin' and 'resolution'"
        )

    xx, yy = grid_coords
    if xx.ndim == 1 and yy.ndim == 1:
        xx, yy = np.meshgrid(xx, yy, indexing="xy", sparse=False)

    # Check output shape
    if output is not None and ~np.all(output.shape == xx.shape):
        raise ValueError("The output buffer's shape does not match the grid's " "shape")

    points = [shapely.Point(x, y) for x, y in zip(xx.flatten(), yy.flatten(), strict=True)]

    # Prepare geometries in order to optimize computation
    # This methods affects objects inplace
    _ = [shapely.prepare(polygon) for polygon in polygons]

    # TODO : STRrtree
    if inner_value not in [0, 1]:
        raise ValueError("The argument 'inner_value' must have a binary value " "(0 or 1)")

    if outer_value not in [0, 1]:
        raise ValueError("The argument 'inner_value' must have a binary value " "(0 or 1)")

    if inner_value == outer_value:
        raise ValueError("The argument 'inner_value' and 'outer_value' must be " "different")

    # Here we adopt the shapely convention :
    # - 0 is used for the exterior
    # - 1 is used for the interior
    # A final inversion is performed in case inner_value is 0

    # Init mask
    if output is not None:
        # reset output with zeros (False)
        output[:, :] = 0
        # Reshape to flatten => this will not perform copy for contiguous arrays
        mask = output.reshape(-1)
    else:
        mask = np.zeros(len(points), dtype=bool, order="C")

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
                output[:, :] = output.astype(bool)
        else:
            if inner_value == 0:
                output[:, :] = np.where(output, 0, 1)
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
) -> np.ndarray:
    """Rasterizes a list of polygons onto a grid using `rasterio.features.rasterize`.

    This method implies the definition of an `AffineTransform`. It is defined
    using the origin and resolution as follows:

    .. math::
        A = \\begin{pmatrix}
        a & b & c \\\\
        d & e & f \\\\
        g & h & i
        \\end{pmatrix}
        =
        \\begin{pmatrix}
        resolution[1] & 0. & origin[1] - 0.5 * resolution[1] \\\\
        0. & resolution[0] & origin[0] - 0.5 * resolution[0] \\\\
        0. & 0. & 1.
        \\end{pmatrix}

    Parameters
    ----------
    shape : Tuple[int, int]
        Target raster size as a tuple `(number of rows, number of columns)`.

    origin : Tuple[float, float]
        The coordinates `(row, col)` of the raster's first element (0,0)
        (top-left corner of the top-left pixel) in the image's coordinate
        reference system, which is used by the geometry.

    resolution : Tuple[float, float]
        The grid's pixel size in the units of the geometry's coordinate
        reference system. `(row_resolution, column_resolution)`. Note that
        `rasterio` typically uses positive Y resolution for "up" and negative
        for "down". Here, `resolution[0]` (row resolution) is used for `e`
        (y-scale, usually negative for geospatial data) and `resolution[1]`
        (column resolution) for `a` (x-scale).

    polygons : List[shapely.geometry.Polygon]
        A list of `shapely.geometry.Polygon` objects to rasterize.
        The polygon coordinates must be defined in the same reference
        frame as the image targeted by the raster, considering the scale
        factor (resolution) and a possible shift of its origin.
        The coordinates of the polygons are here supposed to be given
        following the standard (x: column, y: row) order.

    inner_value : int
        The value to use for pixels covered by the interior of the union of
        polygons.

    outer_value : int
        The value to use for pixels representing the exterior of the union of
        polygons.

    output : Optional[np.ndarray], default None
        An optional preallocated NumPy array buffer to store the result.
        If provided, the rasterization is performed in-place into this array.
        Its shape must match the `shape` argument. If `output` is given, its
        `dtype` will be used, and the `dtype` argument should be `None`.
        The method will reset the values of this buffer with `outer_value`
        before populating it.

    dtype : Optional[np.dtype], default None
        Desired NumPy data type for the output raster. This argument is only
        used if `output` is `None` (i.e., a new array needs to be allocated).
        Note that `bool` type is not supported by `rasterio.features.rasterize`
        for `dtype`; `np.uint8` is a common substitute for binary masks.

    Returns
    -------
    np.ndarray
        The binary raster mask as a NumPy array. If an `output` buffer was
        provided, this will be a reference to that same array, modified in-place.

    Raises
    ------
    ValueError
        If neither `output` nor `dtype` is provided, or if both are provided.

    ValueError
        If the `output` buffer's shape does not match the `shape` argument.

    """
    if (output is not None and dtype is not None) or (output is None and dtype is None):
        raise ValueError("You should either provide the output buffer or the" " dtype argument")
    # Check output shape
    if output is not None and shape is not None and ~np.all(output.shape == shape):
        raise ValueError("The output buffer's shape does not match the shape " "argument")

    # RasterIO method is parametrized through an AffineTransform (a, b, c, d, e,
    # f, g, h, i) corresponding to the affine transform matrix :
    #
    # | a b c |
    # | d e f |
    # | g h i |
    #
    # Usually g, h, i are respectively set to 0, 0, 1
    transform = Affine(
        resolution[1],
        0,
        origin[1] - 0.5 * resolution[1],
        0,
        resolution[0],
        origin[0] - 0.5 * resolution[0],
        0,
        0,
        1,
    )

    # We could also use the from_origin method :
    #  transform = rasterio.transform.from_origin(
    #        west=origin[1]-0.5*resolution[1],
    #        north=origin[0]-0.5*resolution[0],
    #        xsize=resolution[1],
    #        ysize=-resolution[0]
    #        )

    kwargs = {
        "shapes": polygons,
        "transform": transform,
        "all_touched": False,
        "default_value": inner_value,
        "fill": outer_value,
    }
    if output is not None:
        # reset output with fill value
        output[:, :] = outer_value
        kwargs["out"] = output
    else:
        kwargs["out_shape"] = (shape[0], shape[1])
        kwargs["dtype"] = dtype
    raster = rasterize(**kwargs)
    return raster
