# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
This is an example to use the grid_resampling method.

Command to run :
PYTHONPATH=${PWD}/../../python python example_grid_resampling.py
From root : PYTHONPATH=${PWD}/python python doc/examples/example_grid_resampling.py
"""
import os
import logging
from pathlib import Path
import tempfile
import sys

import numpy as np
import sys
import rasterio
import matplotlib.pyplot as plt

from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.core.grid.grid_resampling import array_grid_resampling
from gridr.misc.mandrill import mandrill
#ARRAY_IN_SHAPE = (100, 50)
#ARRAY_IN = np.arange(ARRAY_IN_SHAPE, dtype=np.float64)
np.set_printoptions(threshold=sys.maxsize)

def check_interp():
    array_in = np.arange(7*8, dtype=np.float64).reshape(7, 8)
    
    y = np.arange(array_in.shape[0]*2, dtype=np.float64)/2.
    x = np.arange(array_in.shape[0]*3, dtype=np.float64)/3.
    xx, yy = np.meshgrid(x,y)
    print('xx', xx)
    print('yy', yy)
    print('in', array_in)
    out_cubic = array_grid_resampling(
        interp='cubic',
        array_in=array_in,
        grid_row=yy,
        grid_col=xx,
        grid_resolution=(1,1),
        array_out=None,
        array_in_mask=None,
        grid_mask=None,
        array_out_mask=None,
        nodata_out=-1,
        )
    out_nearest = array_grid_resampling(
        interp='nearest',
        array_in=array_in,
        grid_row=yy,
        grid_col=xx,
        grid_resolution=(1,1),
        array_out=None,
        array_in_mask=None,
        grid_mask=None,
        array_out_mask=None,
        nodata_out=-1,
        )
    out_linear = array_grid_resampling(
        interp='linear',
        array_in=array_in,
        grid_row=yy,
        grid_col=xx,
        grid_resolution=(1,1),
        array_out=None,
        array_in_mask=None,
        grid_mask=None,
        array_out_mask=None,
        nodata_out=-1,
        )
    
    print('cubic', out_cubic)
    print('linear', out_linear)
    print('nearest', out_nearest)
    
    print("stop array_grid_resampling")
    
    
    


def my_resampling_identity():
    array_in = mandrill[0].astype(np.float64)
    y = np.arange(mandrill.shape[1], dtype=np.float64)
    x = np.arange(mandrill.shape[2], dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    
    array_out = np.zeros(xx.shape, dtype=np.float64)
    print(xx, yy)
    print(array_out.shape)
    
    print("start array_grid_resampling")
    array_grid_resampling(
        interp='cubic',
        array_in=array_in,
        grid_row=yy,
        grid_col=xx,
        grid_resolution=(1,1),
        array_out=array_out,
        array_in_mask=None,
        grid_mask=None,
        array_out_mask=None,
        nodata_out=None,
        )
    print("stop array_grid_resampling")
    print(array_out.shape)
    
    #array_out.reshape(xx.shape)
    
    # Affichage des deux images côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 ligne, 2 colonnes

    # Affichage de l'image 1
    axes[0].imshow(mandrill[0])
    axes[0].axis('off')  # Désactive les axes autour de l'image
    axes[0].set_title('Image 1')  # Titre de l'image 1

    # Affichage de l'image 2
    axes[1].imshow(array_out)
    axes[1].axis('off')  # Désactive les axes autour de l'image
    axes[1].set_title('Image 2')  # Titre de l'image 2

    # Affiche le tout
    plt.show()
    
def my_resampling_identity_window():
    array_in = mandrill[0].astype(np.float64)
    y = np.arange(mandrill.shape[1], dtype=np.float64)
    x = np.arange(mandrill.shape[2], dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    
    # Lets define a window :
    # - margin left : 10
    # - margin right: 20
    # - margin top: 50
    # - margin bottom: 40
    window = np.array(((50, array_in.shape[0]-40-1), (10, array_in.shape[1]-20-1)))
    shape_out = np.asarray(xx.shape) - np.array((50+40,10+20))
    
    array_out = np.zeros(shape_out, dtype=np.float64)
    print(xx, yy)
    print(array_out.shape)
    
    print("start array_grid_resampling")
    array_grid_resampling(
        interp='cubic',
        array_in=array_in,
        grid_row=yy,
        grid_col=xx,
        grid_resolution=(1,1),
        array_out=array_out,
        win = window,
        array_in_mask=None,
        grid_mask=None,
        array_out_mask=None,
        nodata_out=None,
        )
    print("stop array_grid_resampling")
    print(array_out.shape)
    
    #array_out.reshape(xx.shape)
    
    # Affichage des deux images côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 ligne, 2 colonnes

    # Affichage de l'image 1
    axes[0].imshow(mandrill[0])
    axes[0].axis('off')  # Désactive les axes autour de l'image
    axes[0].set_title('Image 1')  # Titre de l'image 1

    # Affichage de l'image 2
    axes[1].imshow(array_out)
    axes[1].axis('off')  # Désactive les axes autour de l'image
    axes[1].set_title('Image 2')  # Titre de l'image 2

    # Affiche le tout
    plt.show()

def my_resampling_centered_oversampling(res_x, res_y, rgb=True):
    nvar=3
    if rgb:
        array_in = mandrill.astype(np.float64)
    else:
        array_in = mandrill[0].astype(np.float64)
        nvar=1
    
    # We center the grid on the image center
    xc = mandrill.shape[2] // 2
    yc = mandrill.shape[1] // 2
    
    gridsize_x = mandrill.shape[2] // res_x
    gridsize_y = mandrill.shape[1] // res_y
    print('gridsize', gridsize_x, gridsize_y)
    
    # we target roughly the same output size
    x0 = xc - gridsize_x // 2
    y0 = yc - gridsize_y // 2
    print('x0 y0', x0, y0)
    xN = x0 + gridsize_x 
    yN = y0 + gridsize_y
    print('xN yN', xN, yN)
    y = np.linspace(y0, yN, gridsize_y+1, dtype=np.float64)
    x = np.linspace(x0, xN, gridsize_x+1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    
    nrow_out = (xx.shape[0]-1) * res_y + 1
    ncol_out = (xx.shape[1]-1) * res_x + 1
    array_out = np.zeros((nvar, nrow_out, ncol_out), dtype=np.float64)
    print(x0, y0)
    print(xx, yy)
    print(gridsize_x, gridsize_y)
    print(array_out.shape)
    grid_window = None #np.array(((3,10), (2, 5)))
    
    print("start array_grid_resampling")
    array_grid_resampling(
        interp='cubic',
        array_in=array_in,
        grid_row=yy,
        grid_col=xx,
        grid_resolution=(res_y,res_x),
        array_out=array_out,
        array_in_mask=None,
        grid_mask=None,
        array_out_mask=None,
        nodata_out=None,
        win=grid_window,
        )
    print("stop array_grid_resampling")
    print(array_out.shape)
    
    #array_out.reshape(xx.shape)
    
    # Affichage des deux images côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 ligne, 2 colonnes

    if nvar == 1:
        # Affichage de l'image 1
        axes[0].imshow(mandrill[0], cmap='gray')
        axes[0].axis('off')  # Désactive les axes autour de l'image
        axes[0].set_title('Image 1')  # Titre de l'image 1

        # Affichage de l'image 2
        axes[1].imshow(array_out, cmap='gray')
        axes[1].axis('off')  # Désactive les axes autour de l'image
        axes[1].set_title('Image 2')  # Titre de l'image 2
    else:
        # Affichage de l'image 1
        axes[0].imshow(array_in.transpose(1,2,0).astype(np.uint8))
        axes[0].axis('off')  # Désactive les axes autour de l'image
        axes[0].set_title('Image 1')  # Titre de l'image 1

        # Affichage de l'image 2
        axes[1].imshow(array_out.transpose(1,2,0).astype(np.uint8))
        axes[1].axis('off')  # Désactive les axes autour de l'image
        axes[1].set_title('Image 2')  # Titre de l'image 2
        

    # Affiche le tout
    plt.show()

#my_resampling_centered_oversampling(10,10)
#my_resampling_identity_window()
check_interp()

# import numpy as np
# import matplotlib.pyplot as plt

# # Afficher l'image avec matplotlib
# plt.imshow(mandrill[0], cmap='gray', interpolation='nearest')
# plt.axis('off')  # Masquer les axes
# plt.show()