ROOT_DIR = '/gpfs/gpfs0/k.gizatullin/agrofield_project/preprocessed_data_14.05.2020/'
IMAGE_DIR = "output_batches"
ANNOTATION_DIR = "output_shapes"

INFO = {
    "description": "Test field segmentation dataset",
    "url": "TODO: add url",
    "version": "0.0.1",
    "year": 2020,
    "contributor": "TC_406",
    "date_created": None
}

LICENSES = [
    {
        "id": 1,
        "name": "Not specified",
        "url": "exapmle.com"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'UnknownField',
        'supercategory': 'Field',
    }
]

coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}
