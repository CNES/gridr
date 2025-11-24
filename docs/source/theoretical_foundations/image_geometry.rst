.. image_geometry:

==============
Image Geometry
==============

Continuous Image Domain
-----------------------
We consider that the discrete image originates from an underlying continuous signal

.. math::

    F:\Omega \subset \mathbb{R}^{2} \to \mathbb{R}

defined over a rectangular domain :math:`\Omega`.

The coordinates :math:`(x, y) \in \Omega` are continuous spatial coordinates, with :math:`x` increasing to the right and :math:`y` increasing downward.

The function :math:`F` represents the ideal, infinitely-resolved image intensity at any real-valued location in :math:`\Omega`.

The digital image is a sampling of this continuous signal.


Discrete Image
--------------

.. image:: /images/numeric_image.svg


A discrete image :math:`I` of width :math:`W` and height :math:`H` is the finite regularly sampled function from the underlying continuous signal :math:`F`.

.. math::

    I:\{0, ...,H-1  \} \times \{0, ...,W-1  \} \to \mathbb{R}

with elements :math:`I[i,j] \in  \mathbb{R}` called **pixels**.

Pixel model
***********
A discrete coordinate :math:`(i, j)` refers to the pixel at **row index** :math:`i` and **column index** :math:`j`, with :math:`(0, 0)` referring the upper-left pixel.

Each pixel corresponds to a rectangular region :math:`R_{i,j}` in the continuous domain. `GridR` uses the **pixel-centered sampling convention** :

.. math::

    R_{i,j} = \left[  j - \frac{1}{2}, j + \frac{1}{2} \right[ \times \left[  i - \frac{1}{2}, i + \frac{1}{2} \right[

Consequences :

- pixel centers lie on integer coordinates
- pixel edges lie on half-integer coordinates
- the center-to-center spacing is 1 in both directions.


Sampling Grid
*************
To avoid ambiguity between array indices and geometric coordinates we define the **sampling grid** (the sampling lattice) explicitly :

.. math::

    \mathscr{G} = \{ (x,y) \in \mathbb{R}^{2}  | x \in \mathbb{Z}, y \in \mathbb{Z} \}

In the **pixel-centered sampling convention** used in `GridR`, each integer coordinate :math:`(j, i) \in \mathbb{Z}^{2}` corresponds to the center of pixel :math:`(i, j)` in the discrete image :math:`I`. We therefore use the mapping :

.. math::

    (i, j) \longleftrightarrow  (x, y) = (j, i) \in \mathscr{G}

All geometric operations and transformation are expressed in this coordinate system.

Image Footprint
***************
Given image width :math:`W` and height :math:`H`, the continuous domain covered by the discrete image :math:`I` — the *footprint* — is

.. math::

    \Omega_{I} = \left[ - \frac{1}{2}, H + \frac{1}{2} \right] \times \left[ - \frac{1}{2}, W + \frac{1}{2} \right]
