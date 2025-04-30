# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
GridR notebook utils

ex. DOC_BUILD=1 DOC_BUILD_OUTPUT_DIR=output/png jupyter nbconvert --to markdown --execute grid_resampling_001_work.ipynb --output output3/notebook.md

"""
import os
import numpy as np
import uuid 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Label, Arrow, NormalHead, OpenHead, VeeHead, CustomJS, HoverTool, Div
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.models import WheelZoomTool, HoverTool
from bokeh.io.export import export_png

# This variable can be set in the caller environment in order to tell in which
# context the plot methods are called (ie. interactive notebook or automatic
# documentation build
#from selenium import webdriver
#from selenium.webdriver.firefox.options import Options

from IPython.display import Markdown, Image, display

def create_local_firefox_webdriver():
    options = Options()
    options.add_argument("--headless")
    options.binary_location = firefox_bin
    return webdriver.Firefox(options=options) #, executable_path=geckodriver_path)

def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

def in_doc_build():
    return os.environ.get("DOC_BUILD", "0") == "1"

def plot_im(data, win_rect=None, prefix=None):
    """
    """
    ret = None
    height, width = None, None
    if data.ndim == 3:
        _, height, width = data.shape
    elif data.ndim == 2:
        height, width = data.shape
    
    # Case of doc build
    if in_doc_build():
        # In that case some additionnal variables must be set in the calling
        # environment :
        # - DOC_BUILD_FILES_OUTPUT_DIR_PATH
        # - DOC_BUILD_NOTEBOOK_OUTPUT_PATH
        output_dir = os.environ.get("DOC_BUILD_FILES_OUTPUT_DIR_PATH", None)
        if not output_dir:
            raise Exception("The environment variable "
                    "`DOC_BUILD_FILES_OUTPUT_DIR_PATH` must be set")
        output_notebook_path = os.environ.get("DOC_BUILD_NOTEBOOK_OUTPUT_PATH",
                None)
        if not output_notebook_path:
            raise Exception("The environment variable "
                    "`DOC_BUILD_NOTEBOOK_OUTPUT_PATH` must be set")
        
        # Create the output directory if does not exists
        os.makedirs(output_dir, exist_ok=True)
        # Ensure unique name
        unique_name = f"{prefix}.png"
        if not prefix:
            unique_name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        # Full output path
        path = os.path.join(output_dir, unique_name)
        
        if data.ndim == 2:
            ret = mpl_export_gray_static(data=data, win_rect=win_rect,
                    export_name=path)
        else:
            raise NotImplementedError
        
        # Print the mardown string to render the image
        rel_path = os.path.relpath(ret, os.path.dirname(output_notebook_path))
        display(Markdown(f"![{unique_name}]({rel_path})"))
    else:
        if data.ndim == 2:
            ret = bokeh_plot_gray(data=data, win_rect=None, prefix=None)
        else:
            ret = bokeh_plot_rgb(data)
        display(ret)


def bokeh_show_fig(fig, prefix=None):
    """
    A method to be called in place of the usual show.
    The behaviour of this method differs given the calling context :
    - if called from a notebook it just call the bokeh native show method
    - if called in a build doc context (the environment variable DOC_BUILD is 
        set to 1) it exports the figure to a png.

    In the doc build context the following additional environment variables 
    must be set :
    - DOC_BUILD_OUTPUT_DIR : path to the output directory storing the plot
    """
    if in_doc_build():
        output_dir = os.environ.get("DOC_BUILD_FILES_OUTPUT_DIR_PATH", None)
        if not output_dir:
            raise Exception("The environment variable `DOC_BUILD_OUTPUT_DIR` "
                    "must be set")
        
        os.makedirs(output_dir, exist_ok=True)
        unique_name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(output_dir, unique_name)
        print(f'begin {path}')
        export_png(fig, filename=path, webdriver=create_local_firefox_webdriver())
        print('end')
    elif in_notebook():
        show(fig)
    else:
        print("Cannot detect the show mode. Please set the DOC_BUILD environment "
                "variable to 1 if used for documentation build")

def mpl_export_gray_static(data, win_rect=None, export_name=None):
    """
    Exporte une image grayscale statique (style Bokeh) avec gestion des NaN.

    Args:
        data: 2D numpy array
        win_rect: ((row_start, row_end), (col_start, col_end)) rectangle vert optionnel
        export_name: nom du fichier à écrire (PNG). Si None, nom unique.
    """
    height, width = data.shape

    if export_name is None:
        export_name = f"mpl_gray_{uuid.uuid4().hex}.png"

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # Define vmin and vmax
    finite_data = data[np.isfinite(data)]
    vmin, vmax = np.min(finite_data), np.max(finite_data)

    cmap = cm.get_cmap("Greys_r").copy()
    cmap.set_bad(color='lightgray')  # The NaN color

    img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

    if win_rect is not None:
        (row_start, row_end), (col_start, col_end) = win_rect
        rect = patches.Rectangle(
            (col_start, row_start),
            col_end - col_start + 1,
            row_end - row_start + 1,
            linewidth=2,
            edgecolor='green',
            facecolor='green',
            alpha=0.2
        )
        ax.add_patch(rect)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(export_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return export_name
    

def bokeh_plot_gray(data, win_rect=None, prefix=None):
    """Plot gray data, interactivement ou pour export PNG selon BUILD_DOC"""
    height, width = data.shape
    
    source = ColumnDataSource(data=dict(image=[data.tolist()]))

    p = figure(
        width=width,
        height=height,
        x_range=(0, width),
        y_range=(height, 0),
        tools="",
        toolbar_location=None,
        sizing_mode="fixed",
    )
    p.min_border = p.min_border_left = p.min_border_right = 0
    p.min_border_top = p.min_border_bottom = 0
    p.axis.visible = False
    p.grid.visible = False

    p.image(image=[data], x=0, y=0, dw=width, dh=height, palette="Greys256")

    if win_rect is not None:
        ((row_start, row_end), (col_start, col_end)) = win_rect
        p.rect(
            x=(col_start + col_end) / 2,
            y=(row_start + row_end) / 2,
            width=col_end - col_start + 1,
            height=row_end - row_start + 1,
            color="green",
            alpha=0.2,
            line_width=2
        )

    # JS : callback on mouse move
    info = Div(text="Move cursor on image", width=400)
    p.js_on_event("mousemove", CustomJS(args=dict(div=info, src=source), code="""
        const x = Math.floor(cb_obj.x);
        const y = Math.floor(cb_obj.y);

        const image = src.data.image[0];
        const nx = image[0].length;
        const ny = image.length;

        if (x >= 0 && x < nx && y >= 0 && y < ny) {
            const value = image[y][x];
            div.text = `Pixel (x=${x}, y=${y}) → value : ${value.toFixed(3)}`;
        } else {
            div.text = "No data";
        }
    """))
    bokeh_show_fig(column(row(p.toolbar, p), info))
    return None


def bokeh_plot_rgb(data):
    """Plot rgb data
    
    Args:
        data: 3d numpy array (channel, height, width)
    """

    channels, height, width = data.shape
    
    img = np.empty((height, width), np.uint32)
    view = img.view(dtype=np.uint8).reshape((height, width, 4))
    for i in range(height):
        for j in range(width):
            view[i, j, 0] = np.uint8(data[0, i, j])
            view[i, j, 1] = np.uint8(data[1, i, j])
            view[i, j, 2] = np.uint8(data[2, i, j])
            view[i, j, 3] = 255
    
    rgb_data_list = img.tolist()     # Convertir en liste de listes
    source = ColumnDataSource(data=dict(image=[rgb_data_list]))


    info = Div(text="Move cursor on image", width=400)
    
    p = figure(
        width=width,
        height=height,
        x_range=(0, width),
        y_range=(height, 0),  # Inverser l'axe Y
        tools="pan,wheel_zoom,box_zoom,reset",
        title="Image RGB avec curseur dynamique",
        sizing_mode="fixed",
    )

    p.image_rgba(image=[img], x=0, y=0, dw=width, dh=height)

    # JS : callback on mouse move
    p.js_on_event("mousemove", CustomJS(args=dict(div=info, src=source), code="""
        const x = Math.floor(cb_obj.x);
        const y = Math.floor(cb_obj.y);

        const image = src.data.image[0];
        const nx = image[0].length;
        const ny = image.length;

        if (x >= 0 && x < nx && y >= 0 && y < ny) {
            const pixel = image[y][x];  // [R, G, B]
            
            const r = (pixel >> 24) & 0xFF;
            const g = (pixel >> 16) & 0xFF;
            const b = (pixel >> 8)  & 0xFF;
            const a = (pixel)       & 0xFF;
            div.text = `Pixel (x=${x}, y=${y}) → RGB : (${r.toFixed(3)}, ${g.toFixed(3)}, ${b.toFixed(3)})`;
        } else {
            div.text = "No data";
        }
    """))

    bokeh_show_fig(column(p, info))
    return None
    
