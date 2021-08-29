import os
from pathlib import Path
import pandas as pd
import pydicom
from tqdm import tqdm

from src.dataset.brats import SEQUENCE_TYPES, IMAGING_PLANES
from src.dataset.data_io import get_patients_data, get_patient_slice_paths
from src.dataset.dataset_util import get_base_folder, get_patient_dirs


def get_meta(path: Path, dataset='train') -> pd.DataFrame:
    """
    Returns dataframe:
        index: BraTS21ID
            patient id
        columns:
            - seq_type
                T1w, T1wCE, T2w, or FLAIR
            - width
                width of image
            - height
                height of image
            - seq_length
                Number of frames in sequence
            - image_plane
                orientation of patient
                axial, coronal, sagittal
            - direction
                direction of scan
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        res = []
        base_folder = get_base_folder(dataset_type=dataset)
        patients_dirs = get_patient_dirs(base_folder=base_folder)

        patient_data = get_patients_data(base_folder=base_folder,
                                         patient_dirs=patients_dirs,
                                         dataset_type=dataset)
        for p in tqdm(patient_data):
            for seq in SEQUENCE_TYPES:
                patient_slice_paths = get_patient_slice_paths(
                    patient=p, modality=seq)
                data = pydicom.dcmread(patient_slice_paths[0])
                orientation_patient = data.ImageOrientationPatient
                image_plane = \
                    _get_image_plane(
                        image_orientiation_patient=orientation_patient)
                direction = _get_direction(
                    first_slice_path=patient_slice_paths[0],
                    last_slice_path=patient_slice_paths[-1],
                    plane=image_plane
                )
                res.append({
                    'BraTS21ID': p.id,
                    'seq_type': seq,
                    'width': data.Rows,
                    'height': data.Columns,
                    'seq_length': len(patient_slice_paths),
                    'image_plane': image_plane,
                    'direction': direction
                })
        df = pd.DataFrame(res).set_index('BraTS21ID')
    return df


# https://www.kaggle.com/davidbroberts/determining-mr-image-planes
def _get_image_plane(image_orientiation_patient: list):
    x1, y1, _, x2, y2, _ = [round(j) for j in
                            image_orientiation_patient]
    cords = [x1, y1, x2, y2]

    if cords == [1, 0, 0, 0]:
        return 'coronal'
    if cords == [1, 0, 0, 1]:
        return 'axial'
    if cords == [0, 1, 0, 0]:
        return 'sagittal'


# https://www.kaggle.com/ren4yu/normalized-voxels-align-planes
# -and-crop
def _get_direction(first_slice_path: Path, last_slice_path: Path,
                   plane: str):
    first_slice = pydicom.dcmread(first_slice_path)
    last_slice = pydicom.dcmread(last_slice_path)

    positions = [
        first_slice.ImagePositionPatient,
        last_slice.ImagePositionPatient
    ]

    first_pos_x, first_pos_y, first_pos_z = positions[0]
    last_pos_x, last_pos_y, last_pos_z = positions[-1]

    # in DICOM:
    # x increases from right to left
    # y increases from anterior to posterior
    # z increases from inferior to superior

    if plane == 'coronal':
        if first_pos_y < last_pos_y:
            direction = 'anterior to posterior'
        else:
            direction = 'posterior to anterior'
    elif plane == 'sagittal':
        if first_pos_x < last_pos_x:
            direction = 'right to left'
        else:
            direction = 'left to right'
    elif plane == 'axial':
        if first_pos_z < last_pos_z:
            direction = 'inferior to superior'
        else:
            direction = 'superior to inferior'
    else:
        raise ValueError(f'Expected plane to be in '
                         f'{IMAGING_PLANES}')
    return direction


def main():
    meta = get_meta(path=get_base_folder(
        dataset_type='train') / 'meta.csv')
    meta.to_csv('train_meta.csv')


if __name__ == '__main__':
    main()
