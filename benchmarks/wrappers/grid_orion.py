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
benchmarks/wrappers/grid_orion.py

Wrapper to run CNES Orion legacy resampling tool

The `orion` binary path must be set through the ORION_BIN_PATH environment
variable.
The orion init only script path must be set through the ORION_INIT_ONLY_PATH
environment variable
E.g. :
ORION_BIN_PATH=/work/softs/projets/oc/oc_pack_RH7_20180125_sif_wrappers/bin/orion
ORION_BIN_PATH=/work/ARTEMIS/cots_delivery/icc_co3d/qtispack_minipack_from_sif/bin/orion.sh
"""
import os
import subprocess
from pathlib import Path
from typing import Optional


def grid_orion_init_only():
    """Run the init only script for orion.
    This function is used in order to estimate the time required by the
    initialization of the shell environnment.
    It uses the environment variable ORION_INIT_ONLY_PATH.
    """
    # ------------------------------------------------------------------
    # Check Orion availability
    # ------------------------------------------------------------------
    orion_executable = None
    try:
        orion_executable = os.environ.get("ORION_INIT_ONLY_PATH")
    except KeyError:
        orion_executable = None
    if orion_executable is None:
        raise EnvironmentError("ORION_INIT_ONLY_PATH environment variable is not set")
    return subprocess.run(
        orion_executable,
        shell=True,
        executable="/bin/bash",
        check=False,
        capture_output=True,
        text=True,
    )


def grid_orion(  # noqa C901
    input_image: str,
    output_image: str,
    grid_orion: str,
    surech_row: float,
    surech_col: float,
    origin_row: float,
    origin_col: float,
    nb_row_out: int,
    nb_col_out: int,
    format_grid: str,
    filter: str,  # noqa: A002
    edges_management: str,
    format_in: str = "AUTO",
    format_out: Optional[str] = None,
    type_in: Optional[str] = None,
    type_out: Optional[str] = None,
    slope: float = 1.0,
    bias: float = 0.0,
    no_data: float = 0.0,
    mode_read: str = "IMAGE",
    complex_image: bool = False,
    num_canal_in: Optional[int] = None,
    tile_orion_auto: bool = False,
    largeur_imagette: Optional[int] = None,
    hauteur_imagette: Optional[int] = None,
    taille_cache_image: Optional[int] = None,
    zu_orion_target_in: Optional[str] = None,
    gene_grid_dense: Optional[str] = None,
    use_invalid_data: bool = False,
    invalid_data_val: Optional[float] = None,
    alpha_filter: Optional[float] = None,
    name_filter: Optional[str] = None,
    center_filter_row: Optional[int] = None,
    center_filter_col: Optional[int] = None,
    a_filter: Optional[int] = None,
    sigma_filter: Optional[float] = None,
    precision_filter: Optional[int] = None,
    spline_order: Optional[int] = None,
    antialiasing: bool = False,
    sensor: bool = False,
    name_sensor: Optional[str] = None,
    pente: Optional[float] = None,
    repli: Optional[float] = None,
    nb_processus: int = 1,
    trace: bool = False,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Run the ``orion --grid_orion`` geometric transformation pipeline.

    Applies a grid-based geometric transformation (e.g. rotation, reprojection)
    to a source image using the BibOrion ``--grid_orion`` pipeline.

    The `orion` binary path must be set through the ORION_BIN_PATH environment
    variable

    Parameters
    ----------
    input_image : str
        Path to the input image (``<ENTREE>``).
    output_image : str
        Path to the output image (``<SORTIE>``).
    grid_orion : str
        Path to the grid file (``--grid_orion``).
    surech_row : float
        Oversampling factor applied to the grid along rows (``--surech_row``).
        Must be strictly positive.
    surech_col : float
        Oversampling factor applied to the grid along columns (``--surech_col``).
        Must be strictly positive.
    origin_row : float
        Row position within the grid of the target image origin
        (``--origin_row``). Must be >= 0.
    origin_col : float
        Column position within the grid of the target image origin
        (``--origin_col``). Must be >= 0.
    nb_row_out : int
        Number of rows in the target image (``--nb_row_out``). Must be > 0.
    nb_col_out : int
        Number of columns in the target image (``--nb_col_out``). Must be > 0.
    format_grid : str
        Format of the grid file (``--format_grid``).
        One of ``BSQ``, ``GESSIMU``, ``LUM``, ``SP5LIB``, ``HDF``.
    filter : str
        Radiometric interpolation filter (``--filter``).
        One of ``PPV``, ``BLN``, ``BCC``, ``BCO``, ``BCG``, ``BCGT``,
        ``SCAB``, ``SCA``, ``SCAT``, ``SCATB``, ``SPLINE``.
    edges_management : str
        Border management mode for the source image (``--edges_management``).
        One of ``MIROIR``, ``EXACTE``.
    format_in : str, optional
        Format of the input image (``--format_in``). Default is ``"AUTO"``.
    format_out : str, optional
        Format of the output image (``--format_out``). Inferred from input
        when ``None``.
    type_in : str, optional
        Pixel encoding type of the input image (``--type_in``). Inferred from
        image structure when ``None``.
    type_out : str, optional
        Pixel encoding type of the output image (``--type_out``). Defaults to
        the input type when ``None``.
    slope : float, optional
        Global gain applied to pixel values (``--slope``). Default is ``1.0``.
    bias : float, optional
        Global offset applied to pixel values (``--biais``). Default is ``0.0``.
    no_data : float, optional
        Fill value for target pixels that fall outside the source image
        (``--no_data``). Default is ``0.0``.
    mode_read : str, optional
        File management mode (``--mode_read``).
        One of ``IMAGE``, ``BANDEAU``, ``TUILE``. Default is ``"IMAGE"``.
    complex_image : bool, optional
        Flag indicating the image is complex (``--complex``).
        Default is ``False``.
    num_canal_in : int, optional
        Band number to process (``--num_canal_in``). All bands are processed
        when ``None``.
    tile_orion_auto : bool, optional
        Automatically compute the file management mode
        (``--tile_orion_auto``). Default is ``False``.
    largeur_imagette : int, optional
        Width in pixels of tiles / strips (``--largeur_imagette``).
        Default is ``1000`` on the orion side when ``None``.
    hauteur_imagette : int, optional
        Height in pixels of tiles / strips (``--hauteur_imagette``).
        Default is ``1000`` on the orion side when ``None``.
    taille_cache_image : int, optional
        Cache size in bytes for file management (``--taille_cache_image``).
        Default is ``1 000 000 000`` on the orion side when ``None``.
    zu_orion_target_in : str, optional
        Path to the useful-zone description file of the target image
        (``--zu_orion_target_in``).
    gene_grid_dense : str, optional
        Path for writing the dense grid corresponding to the transformation
        (``--gene_grid_dense``).
    use_invalid_data : bool, optional
        Activate an invalidity value (``--use_invalid_data``).
        Default is ``False``.
    invalid_data_val : float, optional
        Invalidity value to use (``--invalid_data_val``).
        Required when *use_invalid_data* is ``True``.
    alpha_filter : float, optional
        Optimisation coefficient for a general bicubic filter
        (``--alpha_filter``). Required for filters ``BCG``, ``BCGT``,
        ``BCGS``.
    name_filter : str, optional
        Path to an external filter file without extension
        (``--name_filter``). Required for filters ``SCAB``, ``SCAS``.
    center_filter_row : int, optional
        Hot-spot row position of the external filter
        (``--center_filter_row``).
    center_filter_col : int, optional
        Hot-spot column position of the external filter
        (``--center_filter_col``).
    a_filter : int, optional
        Filter support extent (``--A_filter``). Used with filter ``SCA``.
        Default is ``6`` on the orion side when ``None``.
    sigma_filter : float, optional
        Standard deviation of the Gaussian in the frequency domain
        (``--sigma_filter``). Used with filter ``SCA``.
        Default is ``0.068`` on the orion side when ``None``.
    precision_filter : int, optional
        Filter precision (``--precision_filter``). Used with filters
        ``BCGT``, ``SCAT``, ``SCATB``.
    spline_order : int, optional
        Spline order (``--spline_order``). Used with filter ``SPLINE``.
        One of ``3``, ``5``, ``7``.
    antialiasing : bool, optional
        Apply anti-aliasing pre-processing (``--antialiasing``).
        Default is ``False``.
    sensor : bool, optional
        Use a specific MTF for source image filtering (``--sensor``).
        Requires *antialiasing* to be ``True``.
    name_sensor : str, optional
        Path to the MTF file for source image filtering
        (``--name_sensor``). Required when *sensor* is ``True``.
    pente : float, optional
        Slope of the tanh filter for source image filtering
        (``--pente``). Used with *antialiasing*. Default is ``10.0``
        on the orion side when ``None``.
    repli : float, optional
        Spectral fold-back value for source image filtering
        (``--repli``). Used with *antialiasing*. Default is ``0.0``
        on the orion side when ``None``.
    nb_processus : int, optional
        Number of parallel processes (``--nb_processus``). Default is ``1``.
    trace : bool, optional
        Print execution trace (``--trace``). Default is ``False``.

    Returns
    -------
    subprocess.CompletedProcess
        The result of :func:`subprocess.run`. Check ``returncode`` for
        success (``0``) or failure (non-zero).

    Raises
    ------
    ValueError
        If mandatory numeric constraints are violated or if dependent
        parameters are missing.
    FileNotFoundError
        If *input_image* or *grid_orion* paths do not exist on disk.

    Examples
    --------
    Basic rotation of -45 degrees with anti-aliasing:

    >>> result = grid_orion(
    ... input_image="LENNA.lum",
    ... output_image="LENNA_grid_orion.tif",
    ... grid_orion="grid_LENNA.hd",
    ... surech_row=1.0,
    ... surech_col=1.0,
    ... origin_row=0.0,
    ... origin_col=0.0,
    ... nb_row_out=359,
    ... nb_col_out=359,
    ... format_grid="BSQ",
    ... format_in="AUTO",
    ... format_out="TIFF",
    ... type_out="ENTIER_N_8_BITS",
    ... mode_read="IMAGE",
    ... slope=1.0,
    ... bias=0.0,
    ... no_data=0,
    ... filter="BCO",
    ... edges_management="MIROIR",
    ... antialiasing=True,
    ... trace=True,
    ... )
    >>> result.returncode
    0
    """
    # ------------------------------------------------------------------
    # Check Orion availability
    # ------------------------------------------------------------------
    orion_executable = None
    try:
        orion_executable = os.environ.get("ORION_BIN_PATH")
    except KeyError:
        orion_executable = None
    if orion_executable is None:
        raise EnvironmentError("ORION_BIN_PATH environment variable is not set")

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not Path(input_image).exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")
    if not Path(grid_orion).exists():
        raise FileNotFoundError(f"Grid file not found: {grid_orion}")

    if surech_row <= 0:
        raise ValueError(f"surech_row must be > 0, got {surech_row}")
    if surech_col <= 0:
        raise ValueError(f"surech_col must be > 0, got {surech_col}")
    if origin_row < 0:
        raise ValueError(f"origin_row must be >= 0, got {origin_row}")
    if origin_col < 0:
        raise ValueError(f"origin_col must be >= 0, got {origin_col}")
    if nb_row_out <= 0:
        raise ValueError(f"nb_row_out must be > 0, got {nb_row_out}")
    if nb_col_out <= 0:
        raise ValueError(f"nb_col_out must be > 0, got {nb_col_out}")
    if nb_processus <= 0:
        raise ValueError(f"nb_processus must be > 0, got {nb_processus}")

    if use_invalid_data and invalid_data_val is None:
        raise ValueError("invalid_data_val must be provided when use_invalid_data is True")
    if sensor and not antialiasing:
        raise ValueError("sensor requires antialiasing=True")
    if sensor and name_sensor is None:
        raise ValueError("name_sensor must be provided when sensor is True")

    # ------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------
    cmd = [
        orion_executable,
        input_image,
        output_image,
        f"--grid_orion={grid_orion}",
        # Geometric transformation
        f"--surech_row={surech_row}",
        f"--surech_col={surech_col}",
        f"--origin_row={origin_row}",
        f"--origin_col={origin_col}",
        f"--nb_row_out={nb_row_out}",
        f"--nb_col_out={nb_col_out}",
        f"--format_grid={format_grid}",
        # Image info
        f"--format_in={format_in}",
        f"--mode_read={mode_read}",
        f"--slope={slope}",
        f"--biais={bias}",
        f"--no_data={no_data}",
        # Interpolation
        f"--filter={filter}",
        # Borders
        f"--edges_management={edges_management}",
        # Multiprocessing
        f"--nb_processus={nb_processus}",
    ]

    # Optional image format / type
    if format_out is not None:
        cmd.append(f"--format_out={format_out}")
    if type_in is not None:
        cmd.append(f"--type_in={type_in}")
    if type_out is not None:
        cmd.append(f"--type_out={type_out}")

    # Optional image flags
    if complex_image:
        cmd.append("--complex")
    if num_canal_in is not None:
        cmd.append(f"--num_canal_in={num_canal_in}")
    if tile_orion_auto:
        cmd.append("--tile_orion_auto")
    if largeur_imagette is not None:
        cmd.append(f"--largeur_imagette={largeur_imagette}")
    if hauteur_imagette is not None:
        cmd.append(f"--hauteur_imagette={hauteur_imagette}")
    if taille_cache_image is not None:
        cmd.append(f"--taille_cache_image={taille_cache_image}")
    if zu_orion_target_in is not None:
        cmd.append(f"--zu_orion_target_in={zu_orion_target_in}")
    if gene_grid_dense is not None:
        cmd.append(f"--gene_grid_dense={gene_grid_dense}")

    # Invalidity
    if use_invalid_data:
        cmd.append("--use_invalid_data")
    if invalid_data_val is not None:
        cmd.append(f"--invalid_data_val={invalid_data_val}")

    # Filter-dependent options
    if alpha_filter is not None:
        cmd.append(f"--alpha_filter={alpha_filter}")
    if name_filter is not None:
        cmd.append(f"--name_filter={name_filter}")
    if center_filter_row is not None:
        cmd.append(f"--center_filter_row={center_filter_row}")
    if center_filter_col is not None:
        cmd.append(f"--center_filter_col={center_filter_col}")
    if a_filter is not None:
        cmd.append(f"--A_filter={a_filter}")
    if sigma_filter is not None:
        cmd.append(f"--sigma_filter={sigma_filter}")
    if precision_filter is not None:
        cmd.append(f"--precision_filter={precision_filter}")
    if spline_order is not None:
        cmd.append(f"--spline_order={spline_order}")

    # Anti-aliasing
    if antialiasing:
        cmd.append("--antialiasing")
    if sensor:
        cmd.append("--sensor")
    if name_sensor is not None:
        cmd.append(f"--name_sensor={name_sensor}")
    if pente is not None:
        cmd.append(f"--pente={pente}")
    if repli is not None:
        cmd.append(f"--repli={repli}")

    # Trace
    if trace:
        cmd.append("--trace")

    return subprocess.run(
        " ".join(cmd),
        shell=True,
        executable="/bin/bash",
        check=False,
        capture_output=True,
        text=True,
    )
