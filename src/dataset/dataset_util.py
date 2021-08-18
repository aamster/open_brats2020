import os
from pathlib import Path


def get_base_folder(dataset_type: str):
    is_train = dataset_type in ('train', 'val')
    base_folder = Path(os.environ['BRATS_DATA_ROOT']).resolve()
    base_folder = base_folder / 'train' if is_train else 'test'
    return base_folder


def get_patient_dirs(base_folder):
    """Get all dirs for all patients"""
    patient_dirs = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    return patient_dirs
