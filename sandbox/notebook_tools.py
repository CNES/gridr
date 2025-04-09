import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Label, Arrow, NormalHead, OpenHead, VeeHead, CustomJS, HoverTool, Div
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.models import WheelZoomTool, HoverTool


def plot_gray(data, win_rect):
    """Plot gray data
    
    Args:
        data: 2d numpy array
    """
    height, width = data.shape

    data_list = data.tolist()
    source = ColumnDataSource(data=dict(image=[data_list]))

    info = Div(text="Move cursor on image", width=400)

    p = figure(
        width=width,
        height=height,
        x_range=(0, width),
        y_range=(height, 0),
        #tools="pan,wheel_zoom,box_zoom,reset",
        #tools="pan,box_zoom,reset",
        tools="",
        toolbar_location=None,  # Pas dans la figure
        sizing_mode="fixed",
        )
    p.min_border = 0
    p.min_border_left = 0
    p.min_border_right = 0
    p.min_border_top = 0
    p.min_border_bottom = 0
    p.axis.visible = False
    p.grid.visible = False
    
    p.image(image=[data], x=0, y=0, dw=width, dh=height, palette="Greys256")
    
    # If rectangle coordinates are provided, add a green border rectangle
    if win_rect is not None:
        ((row_start, row_end), (col_start, col_end)) = win_rect
        p.rect(
            x=(col_start + col_end) / 2,
            y=(row_start + row_end) / 2,
            width=col_end - col_start + 1,
            height=row_end - row_start + 1,
            color="green",
            alpha=0.2,  # Transparent fill
            line_width=2
        )

    # JS : callback on mouse move
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

    show(column(row(p.toolbar, p), info))


def plot_rgb(data):
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

    show(column(p, info))
