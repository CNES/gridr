# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.grid.grid_rasterize module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_rasterize.py
"""
import numpy as np
import pytest
import shapely

from gridr.core.grid.grid_rasterize import (
    GridRasterizeAlg,
    ShapelyPredicate,
    grid_rasterize,
    rasterize_polygons_rasterio_rasterize,
    rasterize_polygons_shapely,
)

ALG_RASTERIZE_SHAPELY_COVERS = {
    "alg": GridRasterizeAlg.SHAPELY,
    "kwargs_alg": {"shapely_predicate": ShapelyPredicate.COVERS},
}
ALG_RASTERIZE_SHAPELY_INTERSECTS = {
    "alg": GridRasterizeAlg.SHAPELY,
    "kwargs_alg": {"shapely_predicate": ShapelyPredicate.INTERSECTS},
}
ALG_RASTERIZE_SHAPELY_CONTAINS = {
    "alg": GridRasterizeAlg.SHAPELY,
    "kwargs_alg": {"shapely_predicate": ShapelyPredicate.CONTAINS},
}
ALG_RASTERIZE_RASTERIO_RASTERIZE = {"alg": GridRasterizeAlg.RASTERIO_RASTERIZE, "kwargs_alg": {}}


# Test context using grid_coords
DATAx00_COORDS = (
    (
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]),  # grid_coords
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
    ),
    None,
    None,
    None,  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (4.5, 2.5), (4.5, 6.5), (2.5, 6.5)])],  # geometries
    1e-6,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

# Test context using shape, origin, resolution
DATAx00_SOR = (
    None,  # grid_coords
    (8, 12),
    (0.5, 0.5),
    (1, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (4.5, 2.5), (4.5, 6.5), (2.5, 6.5)])],  # geometries
    1e-6,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx00_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    GridRasterizeAlg.RASTERIO_RASTERIZE: np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
}

# Test origin to (0., 0.)
DATAx01_SOR = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (1, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (4.5, 2.5), (4.5, 6.5), (2.5, 6.5)])],  # geometries
    1e-6,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx01_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    GridRasterizeAlg.RASTERIO_RASTERIZE: np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
}

# Test resolution to (2, 1) with origin (0.5, 0.5)
DATAx02_SOR = (
    None,  # grid_coords
    (8, 12),
    (0.5, 0.5),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    1e-6,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx02_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    GridRasterizeAlg.RASTERIO_RASTERIZE: np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
}

# Test resolution to (2, 1) with origin (0., 0.)
DATAx03_SOR = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    None,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx03_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    GridRasterizeAlg.RASTERIO_RASTERIZE: np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
}


# Test error : all args are set to None
DATAx04_ERROR = (
    None,  # grid_coords
    None,
    None,
    None,  # shape, origin, resolution
    [],  # geometries
    None,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx04_ERROR_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): ValueError,
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,
}

# Test error : coords and SOR are passed
DATAx05_ERROR = (
    (
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]),  # grid_coords
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
    ),
    (8, 12),
    (0.5, 0.5),
    (2, 1),  # shape, origin, resolution
    [],  # geometries
    None,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx05_ERROR_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): ValueError,
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,
}

# Test error : coords and SOR are passed (except origin)
DATAx06_ERROR = (
    (
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5]),  # grid_coords
        np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
    ),
    (8, 12),
    None,
    (2, 1),  # shape, origin, resolution
    [],  # geometries
    None,  # geometries buffer
    None,
    np.uint8,
)  # output, dtype

DATAx06_ERROR_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): ValueError,
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,
}

# Test output usage
DATAx07 = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    None,  # geometries buffer
    np.zeros((8, 12), dtype=np.uint8, order="C"),
    None,
)  # output, dtype

DATAx07_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    GridRasterizeAlg.RASTERIO_RASTERIZE: np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
}

# Test output usage : force error by passing a not None dtype (with output passed)
DATAx08_ERROR = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    None,  # geometries buffer
    np.zeros((8, 12), dtype=np.uint8, order="C"),
    np.uint8,
)  # output, dtype

DATAx08_ERROR_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): ValueError,
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,
}

# Test output usage : force error by passing an output shape that does not corresponds to the
# computation grid
DATAx09_ERROR = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    None,  # geometries buffer
    np.zeros((7, 12), dtype=np.uint8, order="C"),
    None,
)  # output, dtype

DATAx09_ERROR_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): ValueError,
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,
}

# Test output usage : force error by passing neither an output shape nor the out buffer
DATAx10_ERROR = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    None,  # geometries buffer
    None,
    None,
)  # output, dtype

DATAx10_ERROR_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.INTERSECTS): ValueError,
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.CONTAINS): ValueError,
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,
}

# Test output usage - using bool
DATAx11 = (
    None,  # grid_coords
    (8, 12),
    (0.0, 0.0),
    (2, 1),  # shape, origin, resolution
    [shapely.geometry.Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 9.5), (2.5, 9.5)])],  # geometries
    None,  # geometries buffer
    np.zeros((8, 12), dtype=bool, order="C"),
    None,
)  # output, dtype

DATAx11_EXPECTED = {
    (GridRasterizeAlg.SHAPELY, ShapelyPredicate.COVERS): np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ),
    GridRasterizeAlg.RASTERIO_RASTERIZE: ValueError,  # Bool not supported by rasterio rasterize
}


class TestGridRasterize:
    """Test class"""

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            (DATAx00_COORDS, DATAx00_EXPECTED, 6),
            (DATAx00_SOR, DATAx00_EXPECTED, 6),
            (DATAx01_SOR, DATAx01_EXPECTED, 6),
            (DATAx02_SOR, DATAx02_EXPECTED, 6),
            (DATAx03_SOR, DATAx03_EXPECTED, 6),
            (DATAx04_ERROR, DATAx04_ERROR_EXPECTED, 6),
            (DATAx05_ERROR, DATAx05_ERROR_EXPECTED, 6),
            (DATAx06_ERROR, DATAx06_ERROR_EXPECTED, 6),
            (DATAx07, DATAx07_EXPECTED, 6),
            (DATAx08_ERROR, DATAx08_ERROR_EXPECTED, 6),
            (DATAx09_ERROR, DATAx09_ERROR_EXPECTED, 6),
            (DATAx10_ERROR, DATAx10_ERROR_EXPECTED, 6),
            (DATAx11, DATAx11_EXPECTED, 6),
        ],
    )
    @pytest.mark.parametrize(
        "alg_param",
        [
            ALG_RASTERIZE_SHAPELY_COVERS,
            ALG_RASTERIZE_SHAPELY_INTERSECTS,
            ALG_RASTERIZE_SHAPELY_CONTAINS,
        ],
    )
    @pytest.mark.parametrize("inner", [0, 1])
    def test_shapely_rasterize(self, data, expected, alg_param, testing_decimal, inner):
        """Test a regular grid interpolation with mask

        Args:
            data : input data as a tuple containing the shape, origin and resolution tuples
            expected: expected data
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_coords, shape, origin, resolution, polygons, _, output, dtype = data
        # alg_params :
        predicate = alg_param["kwargs_alg"]["shapely_predicate"]
        # expected_mask contains either the nominal output or the exception
        try:
            expected_mask = expected[(GridRasterizeAlg.SHAPELY, predicate)]
        except KeyError:
            # test results not defined for this predicate
            return

        outer = 0 if inner else 1

        mask = None
        expected_dtype = None
        # expected_output_id = None
        if output is not None:
            expected_dtype = output.dtype
            # expected_output_id = id(output.data)
        else:
            expected_dtype = dtype
        try:
            mask = rasterize_polygons_shapely(
                polygons,
                inner,
                outer,
                grid_coords,
                shape,
                origin,
                resolution,
                predicate,
                output,
                dtype,
            )
            if output is not None:
                mask = output
        except Exception as e:
            if isinstance(e, expected_mask):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected_mask, BaseException):
                    raise Exception(
                        f"The test should have raised an exceptionof type {expected_mask}"
                    )
            except TypeError:
                pass
            # Check
            if inner == 1:
                expected_mask_inv = None
                if expected_mask.dtype == bool:
                    expected_mask_inv = np.invert(expected_mask)
                else:
                    expected_mask_inv = np.where(expected_mask, 0, 1)
                expected_mask = expected_mask_inv
            np.testing.assert_array_almost_equal(mask, expected_mask, decimal=testing_decimal)
            assert mask.dtype == expected_dtype
            # if expected_output_id is not None:
            #    assert(id(mask.data) == expected_output_id)

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            (DATAx00_SOR, DATAx00_EXPECTED, 6),
            (DATAx01_SOR, DATAx01_EXPECTED, 6),
            (DATAx02_SOR, DATAx02_EXPECTED, 6),
            (DATAx03_SOR, DATAx03_EXPECTED, 6),
            (DATAx07, DATAx07_EXPECTED, 6),
            (DATAx11, DATAx11_EXPECTED, 6),
        ],
    )
    @pytest.mark.parametrize(
        "alg_param",
        [
            ALG_RASTERIZE_RASTERIO_RASTERIZE,
        ],
    )
    @pytest.mark.parametrize("inner", [0, 1])
    def test_rasterio_rasterize(self, data, expected, alg_param, testing_decimal, inner):
        """Test a regular grid interpolation with mask

        Args:
            data : input data as a tuple containing the shape, origin and resolution tuples
            expected: expected data
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        _, shape, origin, resolution, polygons, geometry_buffer, output, dtype = data
        # alg_params :
        # expected_mask contains either the nominal output or the exception
        expected_mask = expected[GridRasterizeAlg.RASTERIO_RASTERIZE]

        outer = 0 if inner else 1

        mask = None
        expected_dtype = None
        expected_output_id = None
        if output is not None:
            expected_dtype = output.dtype
            expected_output_id = id(output.data)
        else:
            expected_dtype = dtype
        try:
            # apply buffer if not None
            if geometry_buffer is not None:
                polygons = [poly.buffer(distance=geometry_buffer) for poly in polygons]

            mask = rasterize_polygons_rasterio_rasterize(
                shape=shape,
                origin=origin,
                resolution=resolution,
                polygons=polygons,
                inner_value=inner,
                outer_value=outer,
                output=output,
                dtype=dtype,
            )

            if output is not None:
                mask = output
        except Exception as e:
            if isinstance(e, expected_mask):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected_mask, BaseException):
                    raise Exception(
                        f"The test should have raised an exceptionof type {expected_mask}"
                    )
            except TypeError:
                pass
            # Check
            if inner == 1:
                expected_mask_inv = None
                if expected_mask.dtype == bool:
                    expected_mask_inv = np.invert(expected_mask)
                else:
                    expected_mask_inv = np.where(expected_mask, 0, 1)
                expected_mask = expected_mask_inv
            np.testing.assert_array_almost_equal(mask, expected_mask, decimal=testing_decimal)
            assert mask.dtype == expected_dtype

            if False:
                if expected_output_id is not None:
                    assert id(mask.data) == expected_output_id

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            (DATAx00_COORDS, DATAx00_EXPECTED, 6),
            (DATAx00_SOR, DATAx00_EXPECTED, 6),
            (DATAx01_SOR, DATAx01_EXPECTED, 6),
            (DATAx02_SOR, DATAx02_EXPECTED, 6),
            (DATAx03_SOR, DATAx03_EXPECTED, 6),
            (DATAx04_ERROR, DATAx04_ERROR_EXPECTED, 6),
            (DATAx05_ERROR, DATAx05_ERROR_EXPECTED, 6),
            (DATAx06_ERROR, DATAx06_ERROR_EXPECTED, 6),
            (DATAx07, DATAx07_EXPECTED, 6),
            (DATAx08_ERROR, DATAx08_ERROR_EXPECTED, 6),
            (DATAx09_ERROR, DATAx09_ERROR_EXPECTED, 6),
            (DATAx10_ERROR, DATAx10_ERROR_EXPECTED, 6),
            (DATAx11, DATAx11_EXPECTED, 6),
        ],
    )
    @pytest.mark.parametrize(
        "alg_param", [ALG_RASTERIZE_SHAPELY_COVERS, ALG_RASTERIZE_RASTERIO_RASTERIZE]
    )
    @pytest.mark.parametrize("inner", [0, 1])
    def test_grid_rasterize(self, data, expected, alg_param, testing_decimal, inner):
        """Test a regular grid interpolation with mask

        Args:
            data : input data as a tuple containing the shape, origin and resolution tuples
            expected: expected data
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_coords, shape, origin, resolution, polygons, geometry_buffer, output, dtype = data
        # alg_params :
        alg = alg_param["alg"]

        win = None  # TO BE TESTED
        reduce = False

        outer = 0 if inner else 1

        # expected_mask contains either the nominal output or the exception
        if alg == GridRasterizeAlg.SHAPELY:
            try:
                predicate = alg_param["kwargs_alg"]["shapely_predicate"]
                expected_mask = expected[(GridRasterizeAlg.SHAPELY, predicate)]
            except KeyError:
                raise
        elif alg == GridRasterizeAlg.RASTERIO_RASTERIZE:
            try:
                expected_mask = expected[GridRasterizeAlg.RASTERIO_RASTERIZE]
            except KeyError:
                raise

        mask = None
        expected_dtype = None
        expected_output_id = None
        if output is not None:
            expected_dtype = output.dtype
            expected_output_id = id(output.data)
        else:
            expected_dtype = dtype
        try:
            mask = grid_rasterize(
                grid_coords=grid_coords,
                shape=shape,
                origin=origin,
                resolution=resolution,
                win=win,
                inner_value=inner,
                outer_value=outer,
                default_value=0,
                geometry=polygons,
                geometry_buffer_dst=geometry_buffer,
                alg=alg_param["alg"],
                output=output,
                dtype=dtype,
                reduce=reduce,
                **alg_param["kwargs_alg"],
            )
            if output is not None:
                mask = output
        except Exception as e:
            if isinstance(e, expected_mask):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected_mask, BaseException):
                    raise Exception(
                        f"The test should have raised an exceptionof type {expected_mask}"
                    )
            except TypeError:
                pass
            # Check

            if inner == 1:
                expected_mask_inv = None
                if expected_mask.dtype == bool:
                    expected_mask_inv = np.invert(expected_mask)
                else:
                    expected_mask_inv = np.where(expected_mask, 0, 1)
                expected_mask = expected_mask_inv
            np.testing.assert_array_almost_equal(mask, expected_mask, decimal=testing_decimal)
            assert mask.dtype == expected_dtype

            if False:
                if expected_output_id is not None:
                    assert id(mask.data) == expected_output_id
