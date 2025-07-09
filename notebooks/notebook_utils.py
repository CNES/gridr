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
import math

import shapely

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredText
from matplotlib import cm

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Label, Arrow, NormalHead, OpenHead, VeeHead, CustomJS, HoverTool, Div, Title
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

def mpl_plot_wrapper(f):
    """
    prefix has to be in kwargs.
    other parameters must be use by plot
    """ 
    def wrapper(*args, **kwargs):
        ret = None
        prefix = kwargs['prefix']
        try:
            prefix = kwargs['prefix']
        except KeyError:
            prefix = None
            
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
            
            fig = f(*args, **kwargs)
            
            plt.savefig(path, bbox_inches='tight', pad_inches=0.1) # Minimal padding around the whole figure
            plt.close(fig)
            
            # Print the mardown string to render the image
            rel_path = os.path.relpath(path, os.path.dirname(output_notebook_path))
            display(Markdown(f"![{unique_name}]({rel_path})"))
        else:
            fig = f(*args, **kwargs)
            #display(fig)
    return wrapper


def plot_im(data, win_rect=None, prefix=None):
    """
    Displays images. Can handle a single NumPy array or a dictionary of (label, data) pairs
    for side-by-side plotting in documentation build mode (Matplotlib) or interactive mode (Bokeh).

    Args:
        data: A 2D/3D numpy array or a dictionary of {'label': numpy.array}.
        win_rect: Optional ((row_start, row_end), (col_start, col_end)) rectangle to highlight.
        prefix: Prefix for output file names in documentation build mode.
    """
    if isinstance(data, dict):
        # Handle dictionary of images for side-by-side display
        if in_doc_build():
            # Matplotlib: Create a single figure with multiple subplots
            output_dir = os.environ.get("DOC_BUILD_FILES_OUTPUT_DIR_PATH", None)
            if not output_dir:
                raise Exception("The environment variable "
                                "`DOC_BUILD_FILES_OUTPUT_DIR_PATH` must be set")
            output_notebook_path = os.environ.get("DOC_BUILD_NOTEBOOK_OUTPUT_PATH", None)
            if not output_notebook_path:
                raise Exception("The environment variable "
                                "`DOC_BUILD_NOTEBOOK_OUTPUT_PATH` must be set")

            os.makedirs(output_dir, exist_ok=True)
            unique_name = f"{prefix}.png" if prefix else f"multi_plot_{uuid.uuid4().hex[:8]}.png"
            path = os.path.join(output_dir, unique_name)

            # Only supporting 2D gray images for multi-plot static export for simplicity
            # Extend mpl_export_multiple_gray_static if you need RGB here
            all_gray = all(img.ndim == 2 for img in data.values())
            if not all_gray:
                raise NotImplementedError(
                    "Multi-plot static export currently only supports 2D grayscale images."
                )

            ret = mpl_export_multiple_gray_static(data_dict=data, win_rect=win_rect, export_name=path)

            rel_path = os.path.relpath(ret, os.path.dirname(output_notebook_path))
            display(Markdown(f"![{unique_name}]({rel_path})"))

        else:
            # Bokeh: Create a row of interactive plots (as before)
            bokeh_plots = []
            for label, img_data in data.items():
                if img_data.ndim == 2:
                    bokeh_plots.append(bokeh_plot_gray(data=img_data, win_rect=win_rect, title=label))
                elif img_data.ndim == 3:
                    bokeh_plots.append(bokeh_plot_rgb(data=img_data, title=label))
                else:
                    raise ValueError("Unsupported data dimension for image in dictionary.")

            combined_figure = row(*bokeh_plots, spacing=5)
            show(combined_figure)
    else:
        # Handle single image as before
        _plot_single_im(data, win_rect, prefix)

def _plot_single_im(data, win_rect, prefix):
    """Internal helper to plot a single image, respecting doc build mode."""
    ret = None
    height, width = None, None
    if data.ndim == 3:
        _, height, width = data.shape
    elif data.ndim == 2:
        height, width = data.shape
    else:
        raise ValueError(f"Unsupported data dimension: {data.ndim}. Expected 2D or 3D.")

    if in_doc_build():
        output_dir = os.environ.get("DOC_BUILD_FILES_OUTPUT_DIR_PATH", None)
        if not output_dir:
            raise Exception("The environment variable "
                            "`DOC_BUILD_FILES_OUTPUT_DIR_PATH` must be set")
        output_notebook_path = os.environ.get("DOC_BUILD_NOTEBOOK_OUTPUT_PATH", None)
        if not output_notebook_path:
            raise Exception("The environment variable "
                            "`DOC_BUILD_NOTEBOOK_OUTPUT_PATH` must be set")

        os.makedirs(output_dir, exist_ok=True)
        unique_name = f"{prefix}.png" if prefix else f"plot_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(output_dir, unique_name)

        if data.ndim == 2:
            ret = mpl_export_gray_static(data=data, win_rect=win_rect, export_name=path)
        else:
            raise NotImplementedError("Static RGB export not implemented for doc build.")

        rel_path = os.path.relpath(ret, os.path.dirname(output_notebook_path))
        display(Markdown(f"![{unique_name}]({rel_path})"))
    else:
        if data.ndim == 2:
            ret = bokeh_plot_gray(data=data, win_rect=win_rect, title=prefix)
        else:
            ret = bokeh_plot_rgb(data=data, title=prefix)
        show(ret)

def mpl_export_multiple_gray_static(data_dict, win_rect=None, export_name=None, max_cols=4, subplot_width_inches=2, subplot_height_inches=2):
    """
    Exports a single Matplotlib figure containing multiple grayscale images as subplots.

    Args:
        data_dict (dict): Dictionary of {'label': 2D numpy array}.
        win_rect: Optional ((row_start, row_end), (col_start, col_end)) rectangle for each plot.
        export_name: File name to write (PNG). If None, a unique name is generated.
        max_cols (int): Maximum number of columns for the subplot grid.
        subplot_width_inches (float): Desired width of each subplot in inches.
        subplot_height_inches (float): Desired height of each subplot in inches.
    Returns:
        str: The path to the saved image file.
    """
    num_plots = len(data_dict)
    if num_plots == 0:
        raise ValueError("No images provided in data_dict.")

    # Determine grid dimensions
    ncols = min(num_plots, max_cols)
    nrows = math.ceil(num_plots / ncols)

    # Calculate overall figure size based on subplot sizes
    fig_width = ncols * subplot_width_inches
    fig_height = nrows * subplot_height_inches

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), dpi=100, squeeze=False) # squeeze=False ensures axes is always 2D
    axes = axes.flatten() # Flatten the axes array for easy iteration

    if export_name is None:
        export_name = f"mpl_multi_gray_{uuid.uuid4().hex}.png"

    cmap = cm.get_cmap("Greys_r").copy()
    cmap.set_bad(color='lightgray')

    for i, (label, data) in enumerate(data_dict.items()):
        ax = axes[i]

        # Handle vmin/vmax
        finite_data = data[np.isfinite(data)]
        if finite_data.size > 0:
            vmin, vmax = np.min(finite_data), np.max(finite_data)
        else:
            vmin, vmax = 0, 1 # Default range if no finite data

        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title(label, fontsize=8) # Small title font size

        if win_rect is not None:
            (row_start, row_end), (col_start, col_end) = win_rect
            rect = mpatches.Rectangle(
                (col_start, row_start),
                col_end - col_start + 1,
                row_end - row_start + 1,
                linewidth=1, # Thinner line for small plots
                edgecolor='green',
                facecolor='green',
                alpha=0.2
            )
            ax.add_patch(rect)

        ax.axis('off') # Turn off axes for cleaner image display

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=0.5) # Minimal padding between subplots
    plt.savefig(export_name, bbox_inches='tight', pad_inches=0.1) # Minimal padding around the whole figure
    plt.close(fig)

    return export_name


def mpl_export_gray_static(data, win_rect=None, export_name=None):
    """
    Exports a static grayscale image (Matplotlib style) with NaN handling.

    Args:
        data: 2D numpy array
        win_rect: ((row_start, row_end), (col_start, col_end)) optional green rectangle
        export_name: File name to write (PNG). If None, a unique name is generated.
    """
    height, width = data.shape

    if export_name is None:
        export_name = f"mpl_gray_{uuid.uuid4().hex}.png"

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100) # Use original sizing strategy

    finite_data = data[np.isfinite(data)]
    if finite_data.size > 0:
        vmin, vmax = np.min(finite_data), np.max(finite_data)
    else:
        vmin, vmax = 0, 1

    cmap = cm.get_cmap("Greys_r").copy()
    cmap.set_bad(color='lightgray')

    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')

    if win_rect is not None:
        (row_start, row_end), (col_start, col_end) = win_rect
        rect = mpatches.Rectangle(
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

def bokeh_plot_gray(data, win_rect=None, title=None):
    """
    Plots grayscale data interactively using Bokeh.
    """
    height, width = data.shape
    source = ColumnDataSource(data=dict(image=[data.tolist()]))
    p = figure(
        width=width, height=height, x_range=(0, width), y_range=(height, 0),
        tools="", toolbar_location=None, sizing_mode="fixed",
    )
    p.min_border = p.min_border_left = p.min_border_right = 0
    p.min_border_top = p.min_border_bottom = 0
    p.axis.visible = False
    p.grid.visible = False
    if title:
        p.add_layout(Title(text=title, align="center"), "above")
    p.image(image=[data], x=0, y=0, dw=width, dh=height, palette="Greys256")
    if win_rect is not None:
        ((row_start, row_end), (col_start, col_end)) = win_rect
        p.rect(
            x=(col_start + col_end) / 2, y=(row_start + row_end) / 2,
            width=col_end - col_start + 1, height=row_end - row_start + 1,
            color="green", alpha=0.2, line_width=2
        )
    info = Div(text="Move cursor on image")
    p.js_on_event("mousemove", CustomJS(args=dict(div=info, src=source), code="""
        const x = Math.floor(cb_obj.x); const y = Math.floor(cb_obj.y);
        const image = src.data.image[0]; const nx = image[0].length; const ny = image.length;
        if (x >= 0 && x < nx && y >= 0 && y < ny) {
            const value = image[y][x]; div.text = `Pixel (x=${x}, y=${y}) → value : ${value.toFixed(3)}`;
        } else { div.text = "No data"; }
    """))
    return column(p, info)

def bokeh_plot_rgb(data, title=None):
    """
    Plots RGB data interactively using Bokeh.
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
    rgb_data_list = img.tolist()
    source = ColumnDataSource(data=dict(image=[rgb_data_list]))
    info = Div(text="Move cursor on image")
    p = figure(
        width=width, height=height, x_range=(0, width), y_range=(height, 0),
        tools="pan,wheel_zoom,box_zoom,reset",
        title="Image RGB avec curseur dynamique" if not title else None,
        sizing_mode="fixed",
    )
    if title:
        p.add_layout(Title(text=title, align="center"), "above")
    p.image_rgba(image=[img], x=0, y=0, dw=width, dh=height)
    p.js_on_event("mousemove", CustomJS(args=dict(div=info, src=source), code="""
        const x = Math.floor(cb_obj.x); const y = Math.floor(cb_obj.y);
        const image = src.data.image[0]; const nx = image[0].length; const ny = image.length;
        if (x >= 0 && x < nx && y >= 0 && y < ny) {
            const pixel = image[y][x];
            const r = (pixel >> 24) & 0xFF; const g = (pixel >> 16) & 0xFF; const b = (pixel >> 8) & 0xFF;
            div.text = `Pixel (x=${x}, y=${y}) → RGB : (${r.toFixed(0)}, ${g.toFixed(0)}, ${b.toFixed(0)})`;
        } else { div.text = "No data"; }
    """))
    return column(p, info)



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


def plot_convention_grid_mesh(
        shape,
        resolution,
        origin,
        x,
        y,
        geometry=None,
        geometry_origin=None,
        mask=None,
        win=None,
        value_color_alpha_map=None,
        plot_res=60,
        title=None,
        prefix=None):
    if in_doc_build():
        output_dir = os.environ.get("DOC_BUILD_FILES_OUTPUT_DIR_PATH", None)
        if not output_dir:
            raise Exception("The environment variable "
                            "`DOC_BUILD_FILES_OUTPUT_DIR_PATH` must be set")
        output_notebook_path = os.environ.get("DOC_BUILD_NOTEBOOK_OUTPUT_PATH", None)
        if not output_notebook_path:
            raise Exception("The environment variable "
                            "`DOC_BUILD_NOTEBOOK_OUTPUT_PATH` must be set")

        os.makedirs(output_dir, exist_ok=True)
        unique_name = f"{prefix}.png" if prefix else f"plot_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(output_dir, unique_name)

        ret = mpl_plot_convention_grid_mesh(
                shape=shape,
                resolution=resolution,
                origin=origin,
                x=x,
                y=y,
                geometry=geometry,
                geometry_origin=geometry_origin,
                mask=mask,
                win=win,
                value_color_alpha_map=value_color_alpha_map,
                plot_res=plot_res,
                title=title,
                export_name=path)
        
        rel_path = os.path.relpath(ret, os.path.dirname(output_notebook_path))
        display(Markdown(f"![{unique_name}]({rel_path})"))
    else:
        ret = bokeh_plot_convention_grid_mesh(
                shape=shape,
                resolution=resolution,
                origin=origin,
                x=x,
                y=y,
                geometry=geometry,
                geometry_origin=geometry_origin,
                mask=mask,
                win=win,
                value_color_alpha_map=value_color_alpha_map,
                plot_res=plot_res,
                title=title)
        show(ret)


def bokeh_plot_convention_grid_mesh(shape, resolution, origin, x, y, geometry=None, geometry_origin=None, mask=None, win=None, value_color_alpha_map=None, plot_res=60, title=None):
    
    cx, cy = x[0,:], y[:,0]
    pixels = np.asarray([[shapely.geometry.Polygon( [(i-resolution[1]/2, j-resolution[0]/2),
                                      (i-resolution[1]/2, j-resolution[0]/2+resolution[0]),
                                      (i-resolution[1]/2+resolution[1], j-resolution[0]/2+resolution[0]),
                                      (i-resolution[1]/2+resolution[1], j-resolution[0]/2),
                                      (i-resolution[1]/2, j-resolution[0]/2),])
           for i in cx] for j in cy])
    
    p = figure(width=shape[1]*plot_res, height=shape[0]*plot_res)
    
    p.x_range.range_padding = p.y_range.range_padding = 0.5
    p.y_range.flipped = True
    p.scatter(x.flatten(), y.flatten(), size=5, color="black", alpha=0.5, marker='cross', )
    p.scatter(x[0,0], y[0,0], size=6, color="red", alpha=1,marker='cross', )
    geom_x, geom_y = [], []
    [(geom_x.append(list(polygon.exterior.coords.xy[0])), geom_y.append(list(polygon.exterior.coords.xy[1]))) for polygon in pixels.flatten() ]
    p.patches('x', 'y', source = ColumnDataSource(dict(x = geom_x, y = geom_y)), line_color = "grey", line_width = 0.2, fill_color=None)
    
    origin_annotation = Label(x=0, y=0, x_units='data', y_units='data',
                     text='(0, 0)', x_offset=0, y_offset=-resolution[0]/4, text_align="center", text_alpha=0.7, text_color='black', text_baseline='bottom', text_font_size='12px' )
    p.add_layout(origin_annotation)
    vh = VeeHead(size=10, line_color="red", fill_color="red")
    p.add_layout(Arrow(end=vh, x_start=0.0, y_start=0., x_end=1, y_end=0, line_color="red"))
    p.add_layout(Arrow(end=vh, x_start=0.0, y_start=0., x_end=0, y_end=1, line_color="red"))
    
    p.grid.grid_line_width = 0.5
    if title:
        p.add_layout(Title(text=title, align="center"), "above")
    
    if geometry is not None and geometry_origin is not None:
        # Lets add a vector geometry defined by the Polygon and its origin mapping convention towards the grid
        # We want to hereby tell that the grid coordinates (0,0) corresponds to the geometry point (0.5, 0.5)
        
        if not isinstance(geometry, list):
            geometry_list = [geometry]
        else:
            geometry_list = geometry
        
        for geometry_feature in geometry_list:
            # Prepare plot convention
            delta = np.array(geometry_origin) - np.array(origin)
            geom_x, geom_y = [], []
            [(geom_x.append(list((polygon.exterior.coords.xy[0]-delta[1]))),
                geom_y.append(list((polygon.exterior.coords.xy[1]-delta[0]))))
                for polygon in (geometry_feature,) ]

            polygon_geometry = p.patches('x', 'y', source = ColumnDataSource(dict(x = geom_x, y = geom_y)),
                line_color = "green", line_width = 3, fill_color=None, name="polygon_geometry")
    
    if mask is not None and value_color_alpha_map is not None:
        # light up rasterize pixels
        for value, color, alpha in value_color_alpha_map:
            geom_x_inner, geom_y_inner = [], []
            if win is None:
                for j,i in (zip(*np.where(mask==value))):
                    geom_x_inner.append(list(pixels[j,i].exterior.coords.xy[0]))
                    geom_y_inner.append(list(pixels[j,i].exterior.coords.xy[1]))
            else:
                for j,i in (zip(*np.where(mask==value))):
                    geom_x_inner.append(list(pixels[j+win[0,0],i+win[1,0]].exterior.coords.xy[0]))
                    geom_y_inner.append(list(pixels[j+win[0,0],i+win[1,0]].exterior.coords.xy[1]))
            raster_patch_inner = p.patches('x', 'y', source = ColumnDataSource(dict(x = geom_x_inner, y = geom_y_inner)),
                line_color = color, line_width = 0.5, fill_color=color, fill_alpha=alpha, name="raster_patch")

    return p


def mpl_plot_convention_grid_mesh(shape, resolution, origin, x, y, geometry=None, geometry_origin=None, mask=None, win=None, value_color_alpha_map=None, plot_res=60, title=None, export_name=None):
    """
    Equivalent Matplotlib plotting function to visualize grid conventions and masks.

    Parameters
    ----------
    shape : tuple
        Shape of the grid (rows, cols).
    resolution : tuple
        Resolution of the grid (row_res, col_res).
    origin : tuple
        Origin of the grid's coordinate system (x_origin, y_origin).
    x : numpy.ndarray
        2D array of x-coordinates for pixel centers.
    y : numpy.ndarray
        2D array of y-coordinates for pixel centers.
    geometry : shapely.geometry.Polygon or list of shapely.geometry.Polygon, optional
        Vector geometry to plot.
    geometry_origin : tuple, optional
        Origin of the geometry's coordinate system (x_origin, y_origin).
    mask : numpy.ndarray, optional
        Raster mask to visualize.
    win : tuple or numpy.ndarray, optional
        Window for the mask display, e.g., ((row_start, row_end), (col_start, col_end)).
    value_color_alpha_map : list of tuples, optional
        Mapping of mask values to (color, alpha) for plotting.
        Example: [(value1, 'red', 0.5), (value2, 'blue', 1.0)]
    plot_res : int, optional
        Resolution factor for plot size (ignored for actual plot scale,
        kept for parameter compatibility).
    title : str, optional
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object.
    matplotlib.axes.Axes
        The Matplotlib Axes object.
    """

    if export_name is None:
        export_name = f"mpl_convention_grid_mesh_{uuid.uuid4().hex}.png"
    
    fig, ax = plt.subplots(figsize=(shape[1] * plot_res / 100, shape[0] * plot_res / 100)) # Adjust figsize for similar visual scale

    # Extract pixel center coordinates
    cx, cy = x[0, :], y[:, 0]
    pixels = np.asarray([[shapely.geometry.Polygon( [(i-resolution[1]/2, j-resolution[0]/2),
                                      (i-resolution[1]/2, j-resolution[0]/2+resolution[0]),
                                      (i-resolution[1]/2+resolution[1], j-resolution[0]/2+resolution[0]),
                                      (i-resolution[1]/2+resolution[1], j-resolution[0]/2),
                                      (i-resolution[1]/2, j-resolution[0]/2),])
            for i in cx] for j in cy])

    # Generate pixel polygons
    pixels_patches = [ mpatches.Polygon(np.array(poly.exterior.coords)) for poly in pixels.flatten() ]
    
    # Plot pixel boundaries
    pixel_collection = PatchCollection(pixels_patches, facecolor='none', edgecolor='grey', linewidth=0.2)
    ax.add_collection(pixel_collection)

    # Plot pixel centroids (x, y are already in the correct "plot" orientation if x is columns, y is rows)
    ax.scatter(x.flatten(), y.flatten(), s=5, color="black", alpha=0.5, marker='x', label="Pixel Centroids")
    ax.scatter(0, 0, s=30, color="red", alpha=1, marker='x', zorder=5 ) # Zorder to ensure visibility

    # Set axis limits
    margin = 2
    ax.set_xlim(x.min() - resolution[1] - margin, x.max() + resolution[1] + margin)
    ax.set_ylim(y.max() + resolution[0] + margin, y.min() - resolution[0] - margin) # Invert y-axis to match "x is downwards" convention

    # Add origin annotation
    ax.annotate(f"(0, 0)", xy=(0, 0), xytext=(0, - resolution[0]/8)
                ,fontsize=10, color='black', alpha=0.7, ha='left', va='bottom')
    
    # Plot coordinate axes (using plot's x,y for direction)
    ax.arrow(0, 0, 0.9, 0, head_width=0.2, head_length=0.1, fc='red', ec='red', zorder=10) # X-axis arrow (right)
    ax.arrow(0, 0, 0, 0.9, head_width=0.2, head_length=0.1, fc='red', ec='red', zorder=10) # Y-axis arrow (down)

    ax.set_aspect('equal', adjustable='box')
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.2)

    # Plot geometry
    if geometry is not None and geometry_origin is not None:
        if not isinstance(geometry, list):
            geometry_list = [geometry]
        else:
            geometry_list = geometry

        for geom_feature in geometry_list:
            # Prepare plot convention
            delta_plot = np.array(geometry_origin) - np.array(origin)

            if isinstance(geom_feature, shapely.geometry.Polygon):
                # Shapely polygons have exterior.coords.xy[0] for x and [1] for y
                # x coordinates (columns in original grid sense)
                geom_x_coords = np.array(geom_feature.exterior.coords.xy[0]) - delta_plot[1] # subtract y-component of delta from shapely x
                # y coordinates (rows in original grid sense)
                geom_y_coords = np.array(geom_feature.exterior.coords.xy[1]) - delta_plot[0] # subtract x-component of delta from shapely y
                ax.plot(geom_x_coords, geom_y_coords, color="green", linewidth=3, zorder=3)
            elif isinstance(geom_feature, shapely.geometry.MultiPolygon):
                for poly in geom_feature.geoms:
                    geom_x_coords = np.array(poly.exterior.coords.xy[0]) - delta_plot[1]
                    geom_y_coords = np.array(poly.exterior.coords.xy[1]) - delta_plot[0]
                    ax.plot(geom_x_coords, geom_y_coords, color="green", linewidth=3, zorder=3)


    # Plot raster mask
    if mask is not None and value_color_alpha_map is not None:
        mask_patches = []
        for value, color, alpha in value_color_alpha_map:
            if win is None:
                rows, cols = np.where(mask == value)
                for j, i in zip(rows, cols):
                    # pixels[j,i] gives the shapely polygon for that pixel
                    # The polygon coordinates are already relative to the grid origin
                    poly = shapely.geometry.Polygon([(coord[0], coord[1]) for coord in pixels[j,i].exterior.coords])
                    mask_patches.append(mpatches.Polygon(np.array(poly.exterior.coords)))
            else:
                # If window is defined, mask corresponds to the windowed subgrid
                # Need to map back to full grid indices for `pixels` array
                win_row_start, win_row_end = win[0, 0], win[0, 1]
                win_col_start, win_col_end = win[1, 0], win[1, 1]

                rows_mask, cols_mask = np.where(mask == value)
                for j_mask, i_mask in zip(rows_mask, cols_mask):
                    # Calculate original grid index
                    j_orig = j_mask + win_row_start
                    i_orig = i_mask + win_col_start

                    poly = shapely.geometry.Polygon([(coord[0], coord[1]) for coord in pixels[j_orig,i_orig].exterior.coords])
                    mask_patches.append(mpatches.Polygon(np.array(poly.exterior.coords)))

            if mask_patches: # Only add collection if there are patches for this value
                raster_collection = PatchCollection(mask_patches, facecolor=color, edgecolor=color, linewidth=0.5, alpha=alpha, zorder=1)
                ax.add_collection(raster_collection)
                mask_patches = [] # Reset for next value
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(export_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return export_name
