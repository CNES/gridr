"""
Convolution with Astridz
"""
from argparse import ArgumentParser
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
from typing import Union, Dict, NoReturn, Tuple

from artemis_io import artemis_io as aio
from lxml import etree
import numpy as np

# Register artemis io formats for orion filter format support
aio.register_all_formats()


def safe_delete_dir(path: str or Path, min_depth: int = 3):
    """
    Delete a directory recursivelly with prior checks :
    - tests that the directory depth is greater or equal than the min_depth
    
    Args:
        path: path of the directory to delete
        min_depth: minimum depth to authorize the directory deletion
    """
    path = Path(path).resolve()
    if len(path.parents) < min_depth:
        return
    else:
        shutil.rmtree(path.as_posix())


def create_astridz_param(
        param_file_out: str or Path,
        final_coding: str = 'I2',
        block_size_x: int = 1024,
        block_size_y: int = 1024,
        zoom: int = 1,
        output_format: str = 'TIFF') -> NoReturn:
    """
    Create ASTRIDZ parameters XML file.
    Please note the implementation only adress the convolution scenario
    
    Args:
        param_file_out: path for the generated file
        final_coding: code string for the output coding :
            - "I2" : 16 bits integers
            - "FLOA" : simple precision float
        block_size_x : block width used by the convolution algorithm
        block_size_y : block height used by the convolution algorithm
        zoom : zoom factor
        output_format : output image format. (Please see xsd schema for a 
            complete list of available formats)
    """
    xml_str = [
            '<ASTRID_PARAMETERS>',
            '<SPECIFIC_HEADER>',
            '    <TYPE>PARAMETERS</TYPE>',
            '    <APPLICABILITY_DATE>'
                     '2000-01-01T00:00:00.0Z</APPLICABILITY_DATE>',
            '    <VERSION_NUMBER>1</VERSION_NUMBER>',
            '    <STATUS>TEST</STATUS>',
            '</SPECIFIC_HEADER>'
            '<DATA>',
            f'    <SCENARIO>F_CONVOLUTION</SCENARIO>',
            f'    <F_ZOOM_FACTOR>{zoom}</F_ZOOM_FACTOR>',
            '    <F_CONVOLUTION_PARAMETERS>',
            f'        <BLOCK_SIZE_X>{block_size_x}</BLOCK_SIZE_X>',
            f'        <BLOCK_SIZE_Y>{block_size_y}</BLOCK_SIZE_Y>',
            '        <NORMALIZATION_CONST>1</NORMALIZATION_CONST>',
            '    </F_CONVOLUTION_PARAMETERS>',
            '    <OUTPUT_PARAMETERS>',
            '        <OUTPUT_CODING_STANDARD>'
                        'LITTLE_ENDIAN</OUTPUT_CODING_STANDARD>',
            '        <RADIOMETRIC_STRETCHING>0</RADIOMETRIC_STRETCHING>',
            f'        <FINAL_CODING>{final_coding}</FINAL_CODING>',
            '        <MINIMUM_MARGIN>0.0</MINIMUM_MARGIN>',
            '        <MAXIMUM_MARGIN>0.000008</MAXIMUM_MARGIN>',
            '    </OUTPUT_PARAMETERS>',
            '    <IMAGE_PARAMETERS>',
            '        <INPUT_IMAGE_CANAL>1</INPUT_IMAGE_CANAL>',
            '        <OUTPUT_IMAGE_FORMAT>'
                         f'{output_format}</OUTPUT_IMAGE_FORMAT>',
            '    </IMAGE_PARAMETERS>',
            '</DATA>',
            '</ASTRID_PARAMETERS>'
            ]

    with open(param_file_out, 'w') as fp:
        fp.write('\n'.join(xml_str))


def get_filter_size(
        filter_in : Union[str, Path, np.ndarray]) -> Tuple[int, int]:
    """
    Returns the filter size as a tuple containing the number of rows and the
    number of columns of the input filter.
    
    Please note that if filter is given as a path, it must match the
    cnes_orion_filter format. The methods uses artemis_io to open the file.
    
    Args:
        filter_in: input filter. It can be given as a path or numpy array.
    
    Returns:
        a tuple containing the number of rows and the number of columns 
    """
    nrows, ncols = -1, -1
    try:
        nrows, ncols = filter_in.shape
    except AttributeError:
        # Not a numpy array
        if str(filter_in)[-2:].lower() == '.f':
            # Orion filter :
            with aio.open(filter_in, driver='cnes_orion_filter') as ds:
                nrows = ds.height
                ncols = ds.width
        else:
            raise Exception('Unknown filter type')
         
    return nrows, ncols


def get_filter_array(filter_in : Union[str, Path, np.ndarray]):
    """
    Returns filter as array
    
    Please note that if filter is given as a path, it must match the
    cnes_orion_filter format. The methods uses artemis_io to open the file.
    
    Args:
        filter_in: input filter. It can be given as a path or numpy array.
    """
    try:
        filter_in.shape
    except AttributeError:
        # Not a numpy array
        if str(filter_in)[-2:].lower() == '.f':
            # Orion filter :
            with aio.open(filter_in, driver='cnes_orion_filter') as ds:
                filter_array = ds.read(1)
        else:
            raise Exception('Unknown filter type')
            
    else:
        # Filter is a numpy array
        filter_array = filter_in
    return filter_array


def create_astrid_filter(
        filter_in : Union[str, Path, np.ndarray],
        filter_out: str or Path,
        oversampling_row: int = 1,
        oversampling_col: int = 1) -> NoReturn:
    """
    Create an astridz XML filter file from input filter as file or np.ndarray
    
    Args:
        - filter_in : input filter as file path or np.ndarray
        - filter_out : path to the output filter file
        - oversampling_row : the oversampling for rows
        - oversampling_col : the oversampling for col
    """
    filter_type = "SPATIAL"
    filter_array = get_filter_array(filter_in)
    
    filter_array_str = [ ' '.join([str(c) for c in filter_array[row_idx,:]])
            for row_idx in range(filter_array.shape[0])]
    
    filter_str = [
            '<ASTRID_SPATIAL_FILTER>',
            '<SPECIFIC_HEADER>',
            '    <TYPE>FILTER</TYPE>',
            '    <APPLICABILITY_DATE>'
                    '2001-12-17T09:30:47.0Z</APPLICABILITY_DATE>',
            '    <VERSION_NUMBER>1</VERSION_NUMBER>',
            '    <STATUS>TEST</STATUS>',
            '</SPECIFIC_HEADER>',
            '<DATA>',
            '   <FILTER>',
            f'        <FILTER_TYPE>{filter_type}</FILTER_TYPE>',
            '        <FILTER_CENTER>',
            f'        <ROW_INDEX>{1 + filter_array.shape[0]//2}</ROW_INDEX>',
            f'        <COL_INDEX>{1 + filter_array.shape[1]//2}</COL_INDEX>',
            '        </FILTER_CENTER>',
            '        <FILTER_OVERSAMPLING>',
            f'            <ROW_WISE>{oversampling_row}</ROW_WISE>',
            f'            <COL_WISE>{oversampling_col}</COL_WISE>',
            '        </FILTER_OVERSAMPLING>',
            '        <FILTER_DATA>',
            '            <BIDIM_FILTER_DATA>',
            f'                <NB_OF_ROWS>{filter_array.shape[0]}</NB_OF_ROWS>',
            f'                <NB_OF_COLS>{filter_array.shape[1]}</NB_OF_COLS>',
            '                <ORIGIN>1</ORIGIN>',
            '                <ARRAY>',
            '\n'.join([f'                    <ROW>{filter_row}</ROW>'
                    for filter_row in filter_array_str]),
            '                </ARRAY>',
            '            </BIDIM_FILTER_DATA>',
            '        </FILTER_DATA>',
            '    </FILTER>',
            '</DATA>',
            '</ASTRID_SPATIAL_FILTER>']

    with open(filter_out, 'w') as fp:
        fp.write('\n'.join(filter_str))


def compute_optimal_block_size(
        image_nrows: int,
        image_ncols: int,
        filter_nrows: int,
        filter_ncols: int,
        zoom: int = 1) -> Tuple[int, int]:
    """
    Computes the optimal block size from image and filter sizes.
        
    Args:
        image_nrows: number of rows of the input image
        image_ncols: number of columns of the input image
        filter_nrows: number of rows of the filter
        filter_ncols: number of columns of the filter
        zoom: zoom factor that will be used in the convolution scenario
    
    Returns:
        the optimal block sizes as a tuple (size x, size y)
    """
    block_size_col = 2**(1 + int(np.log(2*filter_ncols) / np.log(2)))
    block_size_row = 2**(1 + int(np.log(2*filter_nrows) / np.log(2)))
    if block_size_col > image_ncols*zoom:
        block_size_col = 1 + 2*filter_ncols
    if block_size_row > image_nrows*zoom:
        block_size_row = 1 + 2*filter_nrows
    block_size_col = int(np.floor((block_size_col - filter_ncols) / zoom))
    block_size_row = int(np.floor((block_size_row - filter_nrows) / zoom))
    return block_size_col, block_size_row


def parse_report(report: str or Path) -> Dict:
    """
    Parse the ASTRIDZ report file.
    Returns a dictionnary containing report information :
    - 'error.message' : in case of error, the first error message text.
    
    This method is used in the exception management. Therefore it is only
    interested in the main error message.
    It may be extended to cover a larger scope.
    
    Args:
        report: the report file
    
    Returns:
        dictionnary containing report informations.
    """
    report_info = {}
    root = etree.parse(report)
    if root.find('EXECUTION_REPORT/DIAGNOSTIC').text == 'ERROR':
        try:
            report_info['error.message'] = root.findall(
                    "EXECUTION_REPORT/EXECUTION_LOG_REPORT/ERRORS/MESSAGE"
                    "/MESSAGE_TEXT")[0].text
        except:
            report_info['error.message'] = 'None'
    return report_info


def main(astridz_bin: str or Path,
         image_in: str or Path,
         image_out: str or Path,
         filter_in: Union[np.ndarray, str, Path],
         zoom: int,
         temporary_working_dir: str or Path,
         keep_working_dir: bool = True,
         float_output: bool = True,
         force_squared_block_size=True,) -> int:
    """
    Main function to call ASTRIDZ for a convolution scenario.
    
    The output file extension must be tif or tiff. (Others formats are not
    managed by this method)    
    
    Args:
        astridz_bin : the command used to call ASTRIDZ. Please note that it
                can be a wrapping script.
        image_in : path of the input image
        image_out : path of the output image
        filter_in : path of the input filter. This must be either a cnes orion
                filter file or an already existing Astridz XML spatial filter.
                The list of available format may be extended in the future.
        zoom : the zoom factor used in the convolution scenario
        temporary_working_dir : path to a temporary directory used to write
                astridz traces
        keep_working_dir : boolean parameter to keep temporary_working_dir. If
                an error occurs, the working directory is kept even if this
                parameter is set to False.
        float_output : boolean flag to force output encoding to simple 
                precision float.
        force_squared_block_size : boolean flag to force the block to have same
                size in rows and columns.
    
    Returns:
        the astridz returned code (0 if no errors)
    """
    # Get input and output path and convert them to Path object
    image_in = Path(image_in)
    image_out = Path(image_out)
    filter_in = Path(filter_in)
    temporary_working_dir = Path(temporary_working_dir)
    
    # Set the output encoding
    # By default the output is stored as 16 bits integer, unless the
    # float_output parameter is set to True
    final_coding = 'I2'
    if float_output:
        final_coding = 'FLOA'

    # Set and create the temporary working dir
    temporary_working_dir = Path(temporary_working_dir) / "TRACES_ASTRIDZ"
    os.makedirs(temporary_working_dir, exist_ok=True)
    
    # Manage current working directory and place the current process in the
    # working directory
    cwd = os.getcwd()
    os.chdir(temporary_working_dir)
    os.environ['REP_CURRENT_DIRECTORY'] = os.getcwd()

    # Try to remove the file - if it already exists ASTRDIZ will exit on error
    try:
        os.remove(image_out)
    except FileNotFoundError:
        pass

    # Check the type of the input filter and convert it if necessary :

    # - if the filter is given as an XML file : the method assumes it is
    #   already in the spatial filter format (no check except the extension)
    #   A copy of the filter is performed in the working directory
    # - if the filter is not given as a path : it is assumed that it is given
    #   as a numpy array => a conversion is performed
    # - in any other case a conversion is performed, assuming that the format
    #   is known (currently the code only implements the cnes orion filter format
    #   compatibility)
    do_conv = False
    filter_out = temporary_working_dir / "Filter_SPATIAL.xml"
    try:
        filter_ext = os.path.splitext(filter_in)[-1].lower()
    except TypeError:
        # The filter is probably a numpy array (it should be !)
        do_conv = True
    else:
        if filter_ext != '.xml':
            do_conv = True
    if do_conv:
        create_astrid_filter(filter_in, filter_out)
    else:
        shutil.copy(filter_in, filter_out)

    # compute optimal block size
    filter_in_nrows, filter_in_ncols = get_filter_size(filter_in)
    image_in_nrows, image_in_ncols = -1, -1
    with aio.open(image_in) as ds:
        image_in_nrows = ds.height
        image_in_ncols = ds.width
    block_size_x, block_size_y = compute_optimal_block_size(
            image_nrows = image_in_nrows,
            image_ncols = image_in_ncols,
            filter_nrows = filter_in_nrows,
            filter_ncols = filter_in_ncols,
            zoom = 1)
    
    # check the force_squared_block_size parameter to force same size in both
    # directions
    if force_squared_block_size:
        block_size_x = min((block_size_x, block_size_y))
        block_size_y = block_size_x

    # check the output extension.
    # raises an exception if different from "TIF" or "TIFF"
    extension_out = image_out.as_posix().split('.')[-1].upper()
    if extension_out in ['TIF', 'TIFF']:
        extension_out = 'TIFF'
    else:
        raise Exception(f'Unknown extension out {extension_out}')

    # create astridz param file
    param_file_out = temporary_working_dir / "astridz_param.pts"
    create_astridz_param(
            param_file_out=param_file_out,
            final_coding=final_coding,
            block_size_x=block_size_x,
            block_size_y=block_size_y,
            zoom=zoom,
            output_format=extension_out)

    # create an astridz command file
    report_file = temporary_working_dir / "report.cre"
    cmd_file = temporary_working_dir / "astridz_command.cmd"
    cmd = [ '<ASTRID_COMMAND_FILE>',
            '<INPUT_FILES>',
                '<TECHNICAL_PARAMETERS_FILE>'
                    f'{param_file_out.relative_to(temporary_working_dir)}'
                    '</TECHNICAL_PARAMETERS_FILE>',
                '<DECONVOLUTION_FILTER>'
                    f'{filter_out.relative_to(temporary_working_dir)}'
                    '</DECONVOLUTION_FILTER>',
                f'<INPUT_IMAGE>{image_in}</INPUT_IMAGE>',
                f'<WORKING_FOLDER_PATH>.</WORKING_FOLDER_PATH>',
            '</INPUT_FILES>',
            '<OUTPUT_FILES>',
                '<REPORT_FILE>'
                    f'{report_file.relative_to(temporary_working_dir)}'
                    '</REPORT_FILE>',
                f'<OUTPUT_IMAGE>{image_out}</OUTPUT_IMAGE>',
            '</OUTPUT_FILES>',
            '</ASTRID_COMMAND_FILE>'
            ]
    with open(cmd_file, 'w') as fp:
        fp.write('\n'.join(cmd))

    # Execute the ASTRIDZ command
    astridz_cmd = f'{astridz_bin} {cmd_file}'
    o_proc = subprocess.Popen(astridz_cmd,
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t_output = o_proc.communicate()
    os.chdir(cwd)
    # Parse output ; if EndCode is present and different from 0 then an ASTRIDZ
    # error occurs. Else if EndCode is not present there is another problem.
    stdout = str(t_output[0])
    search_end_code = re.search(
            r"End Code = (?P<end_code>\d+)", stdout, re.MULTILINE)
    endcode = None
    if search_end_code is None:
        # End Code not found
        raise Exception(f"Cannot find End Code in standard output :\n"
                f"stdout : \n{t_output[0]}\n"
                f"stderr : \n{t_output[1]}\n")
    else:
        # End Code found:
        # It should be 0 otherwise an error occured
        endcode = int(search_end_code.group('end_code'))
        if endcode != 0:
            exception_message = (f"ASTRIDZ End Code = {endcode}"
                    f"\nPlease see report : {report_file.as_posix()}"
                    f"\nFirst error message : {parse_report(report_file)['error.message']}"
                    f"\nstdout : {t_output[0]}"
                    f"\nstderr : {t_output[1]}\n")
            raise Exception(exception_message)
        elif not keep_working_dir:
            # not keep working dir
            safe_delete_dir(temporary_working_dir)
    return endcode
            

# USAGE EXAMPLE
# if __name__ == '__main__':
    # main(astridz_bin=Path('/work/ARTEMIS/externals/qtispack_sif/sif_astridz_v5_15.sh'),
         # image_in=Path('/home/il/kelbera/work_campus/artemis/contrib/worker/AstridzImageFilteringWorker_ake/src/artemis_plugins/astridz_image_filtering_worker/resources/scripts/amiens_10_r_6_41_ref_extrait.tif'),
         # image_out=Path('/home/il/kelbera/work_campus/artemis/contrib/worker/AstridzImageFilteringWorker_ake/src/artemis_plugins/astridz_image_filtering_worker/resources/scripts/amiens_10_r_6_41_ref_extrait_conv_dezoom2.tif'),
         # filter_in=Path("/work/CAMPUS/users/kelbera/artemis/contrib/tools/artemis_image_filter/tests_data/in/filter_1_0_0_1_spatial_0_10_128_128_1_1_0.5_0_0_0.5_spatial.f"),
         # 
         # zoom=1,
         # temporary_working_dir="/work/scratch/data/kelbera/tmp_astridz_test",
         # keep_working_dir= True,
         # float_output= True,
         # force_squared_block_size=True)
