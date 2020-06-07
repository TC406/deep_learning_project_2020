from aeronet.dataset import BandCollection, FeatureCollection, rasterize
from aeronet.dataset.raster import Band
from aeronet.converters.split import split
from pathlib import Path
import rasterio
import numpy as np
import aeronet.dataset as ds
from aeronet.dataset.transforms import polygonize
import imageio
import tqdm
import os
from rasterio.windows import Window


def image_quality_pass(scl_mask,
                       contamination_ratio=0.3, contamination_tag_list=[2,3,6,8,11]):
    """
    Check if there are too much contamination on the image like cloud shadows, cloud, water, etc...
    More about contamination tags:
    https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    Args:
        sample_coord: tuple of y, x, heigh, width in pixel.
        scl_mask: rasterio open file.
        contamination_ratio: float. Ratio of what percentage of the image could be contaminated
        contamination_tag_list: list of ints. Type of contamination to take into account.

    Returns:
        bool: True if contamination pixel ratio is smaller than contamination ratio.
              False otherwise.
    """
    contamination_sum = 0
    scl_mask_sample = scl_mask.numpy()
    for contamination_tag in contamination_tag_list:
        contamination_sum += np.sum(scl_mask_sample==contamination_tag)
    if contamination_sum / (scl_mask_sample.shape[1]
                            * scl_mask_sample.shape[2]) > contamination_ratio:
        return False
    else:
        return True


def create_coco_dataset(tiff_filenames, shapes_filename,
                        output_batches_path, output_shapes_path,
                        SCL_path=None,
                        sample_shape=(256, 256),
                        extention="tiff", object_name="UnknownField",
                        contamination_ratio=0.2,
                        start_image_index=1,
                        start_segmentation_index=1,
                        segmentation_are_filter=64,
                        cut_off_b08_channel=[1280, 4800]):
    """
    Slices geotiff image into image batches and create binary mask for instance segmentation
    in COCO-like format.
    Args:
        tiff_filenames: list of strings. Full or relative path to raw image.
        shapes_filename: string. Full or relative path to geojson file with vector fields.
        output_batches_path: string. Full or relative path to directory where image slices
                             would be saved.
        output_shapes_path: string. Full or relative path to directory where binary mask
                            would be saved
        sample_shape: tuple with shapes like (128,128).
        extention: string "tiff" or "png". File type in which image slices will be saved.
                    If "png" is selected then image would be scales to np.uint8 dtype.
        object_name: string. Name of the instance that are in the dataset.

    Returns:
        0
    """
    # tiff_filenames_path = Path(tiff_filenames)
    # for filename_buf in tiff_filenames:
    #     print("Processing files: ", filename_buf)

    with rasterio.open(tiff_filenames[0]) as src:
        profile_mask = src.profile
    profile_mask.update(dtype='uint8', nbits=1, count=1, compress='LZW')
    print("Reading shapes")
    field_collection = FeatureCollection.read(shapes_filename)
    field_collection.crs = rasterio.crs.CRS.from_epsg(32632)
    intersect_field = np.ones(sample_shape, dtype=np.uint8)
    black_ratio = 0.4
    print("Reading first image", tiff_filenames[0])
    bc = split(tiff_filenames[0], tiff_filenames[0].split("/")[-1], ['BLU', 'GRN', 'RED'])

    if len(tiff_filenames) > 1:
        print("Reading second image", tiff_filenames[1])
        for filename in tiff_filenames[1:]:
            bc.append(Band(filename))
    print("prepearing for cycle")
    bands_to_save = [bc._bands[i].name for i in range(len(bc._bands))]

    non_fields = [247,248,249,250,251,252,253,254,255,256,257,259,271,272,273,
                  274,276,277,278,279,285,286,287,305,308,309,310,311,312,313,
                  314,316,317,318,319,320,321,323,324,325,360,361,592,593,594,
                  596,597,602,603,604,605,900,903,905,907,908,920,921,997,580,
                  581,582,583,585,586,587,588,589,590,591]

    if SCL_path is None:
        scl_mask = None
    else:
        scl_mask = Band(SCL_path)
        if scl_mask.shape[-1] != bc.shape[-1]:
            scl_mask = scl_mask.resample(dst_res=(10, 10), interpolation='nearest')
        bc.append(scl_mask)

    scl_layer_name = bc._bands[-1].name
    image_index = start_image_index
    segmentation_index = start_segmentation_index
    samples_used = []
    print("cut cycle started")
    for generated_sample in tqdm.tqdm(bc.generate_samples(sample_shape[0], sample_shape[1])):
        # Right part is number of pixels without data
        # Left part number of pixels that are allowed to be empty
        crs = generated_sample.ordered('BLU','GRN','RED').crs
        transform = generated_sample.ordered('BLU','GRN','RED').transform
        filled_bandsample = ds.BandSample('mask',intersect_field,crs=crs,transform=transform)
        poligonized_bs = polygonize(filled_bandsample)
        if len(field_collection.intersection(poligonized_bs[0])) == 0:
            continue
        if image_quality_pass(generated_sample.ordered(scl_layer_name), contamination_ratio=contamination_ratio):
            if extention is "tiff":
                data_to_write_buf = np.swapaxes(np.swapaxes(generated_sample.ordered(*bands_to_save).numpy(), 1, 2), 2, 0)
                nir_channel_correct = (data_to_write_buf[:,:,3] - cut_off_b08_channel[0]).astype(float)
                nir_channel_correct = nir_channel_correct / (cut_off_b08_channel[1] - cut_off_b08_channel[0]) * 255
                nir_channel_correct = nir_channel_correct.astype(np.uint16)
                data_to_write_buf[:,:,3] = nir_channel_correct
                imageio.imwrite((output_batches_path + "{0:d}." + extention).format(image_index),
                                 data_to_write_buf.astype(np.uint8))
            if extention is "png":
                # Scailing
                buf_image = np.swapaxes(np.swapaxes(generated_sample.ordered('BLU', 'GRN', 'RED').numpy(), 1, 2), 2, 0).copy()
                buf_image = buf_image - buf_image.min(0).min(0)
                buf_image = (buf_image / buf_image.max(0).max(0) * 255)
                imageio.imwrite((output_batches_path + "{0:d}." + extention).format(image_index),
                                buf_image.astype(np.uint8))
            samples_used.append(image_index)
            for intersected_field in field_collection.intersection(poligonized_bs[0]):
                if int(intersected_field.properties['Afgkode']) in non_fields:
                    continue
                mask_to_save = rasterize(FeatureCollection([intersected_field]),
                                         generated_sample.ordered('BLU', 'GRN', 'RED').transform, sample_shape, 'mask0').numpy()
                # Filter small fields
                if np.sum(mask_to_save) > segmentation_are_filter:
                    imageio.imwrite(output_shapes_path + "{0:d}_{1:}_{2:d}.png".format(image_index,
                                                                                       object_name,
                                                                                       segmentation_index),
                                    mask_to_save * 255)
                    segmentation_index += 1

            image_index += 1

def split_dataset(batches_path, shapes_path,
                  split_batches_path, split_shapes_path,
                  object_name="UnknownField",
                  regex_batches_template="([0-9]{1,5}(.)[A-z]{1,5})",
                  regex_shapes_template=["([0-9]{1,5}(_", "_)[0-9]{1,5}(.)[A-z]{1,5})"],
                  validate_ratio=0.3):
    """"""
    # ([0-9]{1,5}(_Unknown_Field_)[0-9]{1,5}(.)[A-z]{1,5})
    # ([0-9]{1,5}(.)[A-z]{1,5})
    # find . ! -iregex ".*\.php.*" -exec cp {} /destination/folder/ \;
    batches_name_list = os.listdir('preprocessed_data/output_batches/')
    index_to_move = np.arange(len(batches_name_list), dtype=int)
    np.random.shuffle(index_to_move)
    for filename in batches_name_list[index_to_move]:
        regex_batches_string = regex_batches_template
        regex_shapes_string = (regex_shapes_template[0] + object_name
                               + regex_shapes_template[1])
        bash_command_batches = ("mv "
                                + batches_path
                                + filename
                                + " "
                                + split_batches_path)

        bash_command_shapes = ("mv "
                               + shapes_path
                               + split_shapes_path
                               + " "
                               + split_batches_path)
        
        print(bash_command_batches, '\n', bash_command_shapes)
    pass