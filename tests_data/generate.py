from pathlib import Path
from typing import Union
import numpy as np
from numpy.lib.stride_tricks import as_strided

import rasterio
from rasterio.windows import Window

def block_view(A, block= (3, 3)):
    shape= (A.shape[0]// block[0], A.shape[1]// block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return as_strided(A, shape= shape, strides= strides)

def create_checkerboard(
        path_out: Path,
        dtype_out: np.dtype,
        value0: Union[int, float],
        value1: Union[int, float],
        rect_size_row: int,
        rect_size_col: int,
        nrect_row: int,
        nrect_col: int,):
    """
    """
    nrow = nrect_row * rect_size_row
    ncol = nrect_col * rect_size_col
    
    arr = np.ones((rect_size_row, ncol), dtype=dtype_out)

    # create a block view
    arr_view = block_view(arr, block=(rect_size_row, rect_size_col))

    with rasterio.open(path_out, "w",
            driver_name="GTiff",
            dtype=dtype_out,
            height=nrow,
            width=ncol,
            count=1,
            ) as ds_out:
        
        for strip in range(nrect_row):
            print(strip)
            v0 = value0
            v1 = value1
            if strip%2:
                v0 = value1
                v1 = value0
            arr_view[0,0::2] = v0
            arr_view[0,1::2] = v1
            
            ds_out.write(arr, 1, window=Window.from_slices(
                    (strip*rect_size_row, (strip+1)*rect_size_row), (0, ncol)))

if __name__ == '__main__':
    cb_v0 = 10
    cb_v1 = 100
    cb_rect_size = (200, 200)
    cb_nrect = (500, 20)
    cb_path_out = f"./checker_board_{cb_v0}_{cb_v1}_{cb_rect_size[0]}_{cb_rect_size[1]}_{cb_nrect[0]}x{cb_nrect[1]}.tif"
    create_checkerboard(path_out=cb_path_out, dtype_out=np.uint8, value0=cb_v0, value1=cb_v1,rect_size_row=cb_rect_size[0], rect_size_col=cb_rect_size[1], nrect_row=cb_nrect[0], nrect_col=cb_nrect[1])