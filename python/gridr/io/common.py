# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Module for common IO definitions
# @doc
"""
from enum import IntEnum

class GridRIOMode(IntEnum):
    """Define the IO mode to consider in certain computaiton (input or output)
    """
    INPUT = 1
    OUTPUT = 2