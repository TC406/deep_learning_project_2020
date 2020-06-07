import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import datetime
import os
import tqdm
from config_coco_json import *

IMAGE_DIR = os.path.join(ROOT_DIR, IMAGE_DIR)
ANNOTATION_DIR = os.path.join(ROOT_DIR, ANNOTATION_DIR)
INFO["date_created"] = datetime.datetime.utcnow().isoformat(' ')


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.tiff', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files


def create_coco_json(json_filename="instances_field.json", verbose=False):
    """
    Creates COCO-json converting png binary mask (0 or 255) to COCO sparse format.
    All args, except for filename should be included in config_coco_json.
    In the end saves result json to drive.
    Args:
        json_filename: str

    Returns:
        None
    """
    image_id = 1
    segmentation_id = 1
    for root,_,files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root,files)
        if verbose:
            print("Number of files: ", len(image_files))

        # go through each image
        for image_filename in tqdm.tqdm(image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(image_id,
                                                              os.path.basename(image_filename),
                                                              image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _ ,files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:

                    # print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id,'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id,image_id,category_info,binary_mask,
                        image.size,tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open(('{}/' + json_filename).format(ROOT_DIR),'w') as output_json_file:
        json.dump(coco_output,output_json_file)
    pass