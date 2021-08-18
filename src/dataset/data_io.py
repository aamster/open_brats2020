import os
from pathlib import Path
from typing import List
import pandas as pd

from src.dataset.patient import Patient


def get_patients_data(base_folder: Path,
                       patient_dirs: List[Path],
                      dataset_type: str) -> List[Patient]:
    """
    Returns list of dict for each patient containing:
        - patient id
        - path to each modality
        - mgmt_val if mode is train
    """
    mgmt_vals = get_mgmt_vals(base_folder=base_folder.parent) \
        if dataset_type in ('train', 'val') else None

    patient_meta = []
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        patient = Patient(
            id=patient_id,
            T1w_path=patient_dir / 'T1w',
            T1wCE_path=patient_dir / 'T1wCE',
            T2w=patient_dir / 'T2w',
            FLAIR_path=patient_dir / 'FLAIR',
            mgmt_val=mgmt_vals.loc[patient_id] if mgmt_vals else None
        )
        patient_meta.append(patient)
    return patient_meta


def get_mgmt_vals(base_folder: Path):
    mgmt_vals = pd.read_csv(base_folder / 'train_labels.csv',
        dtype={'BraTS21ID': str}).set_index('BraTS21ID')
    return mgmt_vals


def get_patient_slice_paths(patient: Patient,
                             modality: str) -> List[Path]:
    path = getattr(patient, f'{modality}_path')
    slices = os.listdir(path)
    images = sorted(slices,
                    key=lambda x: int(Path(x).stem.split('-')[1]))
    im_paths = [Path(f'{path}/{im}') for im in images]
    return im_paths
