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