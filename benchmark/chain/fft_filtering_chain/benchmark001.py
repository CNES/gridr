from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import rasterio
from artemis_io import artemis_io as aio
aio.register_all()

from gridr.chain.fft_filtering_chain import fft_filtering_oa_strip_chain
from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.convolution.fft_filtering import (fft_array_filter_output_shape,
        BoundaryPad, ConvolutionOutputMode)

import astridz_convolution

# Data has to be generated first
# Put here the test data dir
benchmark_data_path = Path(__file__).parent.parent.parent.absolute() / "data"

raster_path_in = benchmark_data_path / "checker_board_10_100_200_200_500x20.tif"
filter_in = benchmark_data_path / "filtre_dezoom2.f"

benchmark_output_dir = Path(".")
raster_os_strip_chain_path_out = benchmark_data_path / "fft_filtering_oa_strip_chain_benchmark01_output_os_strip_chain_out.tif"
raster_os_strip_chain_path_out_2048 = benchmark_data_path / "fft_filtering_oa_strip_chain_benchmark01_output_os_strip_chain_out_2048.tif"
raster_os_strip_chain_path_out_0 = benchmark_data_path / "fft_filtering_oa_strip_chain_benchmark01_output_os_strip_chain_out_0.tif"
raster_astridz_path_out = benchmark_data_path / "fft_filtering_oa_strip_chain_benchmark01_output_astridz_out.tif"

def bench_astridz(raster_path_in, raster_path_out, filter_in):
    d0 = datetime.now()
    print('begin astridz',d0)
    astridz_convolution.main(astridz_bin=Path('/work/ARTEMIS/externals/qtispack_sif/sif_astridz_v5_15.sh'),
         image_in=raster_path_in,
         image_out=raster_astridz_path_out,
         filter_in=filter_in,
         zoom=1,
         temporary_working_dir=Path(benchmark_output_dir).resolve(),
         keep_working_dir= True,
         float_output= False,
         force_squared_block_size=True)
    d1 = datetime.now()
    print('end astridz',d1)
    print(f'astridz processed in {(d1-d0).seconds} sec.')
         

def bench_gridr_oa_strip_chain(raster_path_in, raster_path_out, filter_in, strip_size):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    log_rio = rasterio.logging.getLogger()
    log_rio.setLevel(logging.ERROR)

    # open raster in
    ima_in_ds = rasterio.open(raster_path_in)
    
    # open filter in
    filter_ds = aio.open(filter_in, driver='cnes_orion_filter')
    filter_data = filter_ds.read(1)

    # create expected output shape
    shape_out = (ima_in_ds.height, ima_in_ds.width)
    
    shape_out = fft_array_filter_output_shape(arr=ArrayProfile.from_dataset(ima_in_ds),
            fil=filter_data,
            win=None,
            boundary=BoundaryPad.REFLECT,
            out_mode=ConvolutionOutputMode.SAME,
            zoom = 1,
            axes = None,
            )
    
    with rasterio.open(raster_path_out, "w",
                driver_name="GTiff",
                dtype=np.int16,
                height=shape_out[0],
                width=shape_out[1],
                count=1,) as ds_out:
                #nbits=1) as ds_out:
    
        # Time it here):
        d0 = datetime.now()
        print('begin fft filtering gridr',d0)
        fft_filtering_oa_strip_chain(
                ds_in=ima_in_ds,
                ds_out = ds_out,
                band = 1,
                fil = filter_data,
                boundary = BoundaryPad.REFLECT,
                out_mode = ConvolutionOutputMode.SAME,
                strip_size = strip_size,
                zoom = 1,
                logger=logger,)
        # Time it here
        d1 = datetime.now()
        print('end fft filtering gridr',d1)
        print(f'fft filtering gridr processed in {(d1-d0).seconds} sec.')

if __name__ == '__main__':

    print("benchmark data path", benchmark_data_path)
    
    print("run fft filtering strip 512")
    bench_gridr_oa_strip_chain(raster_path_in=raster_path_in,
            raster_path_out=raster_os_strip_chain_path_out,
            filter_in=filter_in,
            strip_size=512)
    
    print("run fft filtering strip 2048")
    bench_gridr_oa_strip_chain(raster_path_in=raster_path_in,
            raster_path_out=raster_os_strip_chain_path_out_2048,
            filter_in=filter_in,
            strip_size=2048)

    print("run fft filtering strip 0")
    bench_gridr_oa_strip_chain(raster_path_in=raster_path_in,
            raster_path_out=raster_os_strip_chain_path_out_0,
            filter_in=filter_in,
            strip_size=0)
    
    print("run astridz")
    bench_astridz(raster_path_in=raster_path_in, raster_path_out=raster_astridz_path_out, filter_in=filter_in)