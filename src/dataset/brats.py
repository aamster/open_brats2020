from typing import List, Optional, Tuple

import pandas as pd
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
from skimage import io

from src.dataset.data_io import get_patients_data, get_patient_slice_paths
from src.dataset.dataset_util import get_base_folder, get_patient_dirs
from src.dataset.image_utils import pad_or_crop_image, \
    normalize
from src.dataset.patient import Patient

IMAGING_PLANES = ('axial', 'coronal', 'sagittal')
SEQUENCE_TYPES = ('T1w', 'T1wCE', 'FLAIR', 'T2w')


class Brats(Dataset):
    def __init__(self,
                 patient_dirs: List[Path],
                 dataset_type: str,
                 meta: pd.DataFrame,
                 debug=False,
                 normalisation="minmax",
                 limit_to_series_types: Optional[Tuple] = None,
                 normalize_contrast=True):
        """
        @param patient_dirs:
            Path to each patient that should be returned by this dataset
        @param dataset_type:
            train, val or test
        @param meta:
            metadata for dataset
            index of BraTS21ID
        @param debug:
        @param normalisation:
            What normalization method to apply
            minmax or zscore
        @param limit_to_series_types:
            Only use these series types, must be one of SEQUENCE_TYPES
        @param normalize_contrast:
            Clips high and low pixel values to normalize contrast
        """
        dataset_types = ('train', 'val', 'test')
        if dataset_type not in dataset_types:
            raise ValueError(f'dataset type {dataset_type} is not a valid '
                             f'dataset type')
        super(Brats, self).__init__()
        self._dataset_type = dataset_type
        self._patient_dirs = patient_dirs

        base_folder = get_base_folder(dataset_type=dataset_type)
        self.normalisation = normalisation
        self.debug = debug
        self._pad_or_crop_image = True if dataset_type == 'train' else False
        self._patient_data = get_patients_data(
            base_folder=base_folder,
            patient_dirs=self._patient_dirs,
            dataset_type=dataset_type
        )
        self._meta = meta
        self._normalize_contrast = normalize_contrast

        def _validate_limit_to_series_types(series_types):
            for st in series_types:
                if st not in SEQUENCE_TYPES:
                    raise ValueError(
                        f'Series type must be in {SEQUENCE_TYPES}')

        if limit_to_series_types:
            _validate_limit_to_series_types(series_types=limit_to_series_types)
            self._limit_to_series_types = limit_to_series_types
        else:
            # Use all sequence types
            self._limit_to_series_types = SEQUENCE_TYPES

    def __getitem__(self, idx):
        patient = self._patient_data[idx]
        patient_data = self._load_series_for_patient(patient=patient)
        patient_data = [normalize(sequence=x) for x in patient_data]
        mgmt_val = patient.mgmt_val if self._dataset_type in ('train',
                                                              'val') else None

        # TODO WIP

        # # Remove maximum extent of the zero-background to make future crop
        # # more useful
        # z_indexes, y_indexes, x_indexes = np.nonzero(
        #     np.sum(patient_data, axis=0) != 0)
        #
        # # Add 1 pixel on each side
        # zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in
        #                     (z_indexes, y_indexes, x_indexes)]
        # zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in
        #                     (z_indexes, y_indexes, x_indexes)]
        # patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
        # patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        #
        # if self._pad_or_crop_image:
        #     # default to 128, 128, 128
        #     patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        #
        # patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        # patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        # return dict(
        #     patient_id=patient["id"],
        #     image=patient_image,
        #     label=patient_label,
        #     crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
        #     et_present=et_present,
        #     supervised=True)
        return patient_data, mgmt_val

    def _load_series_for_patient(self, patient: Patient) -> List[np.ndarray]:
        res = []
        for series_type in self._limit_to_series_types:

            im_paths = get_patient_slice_paths(patient=patient,
                                               modality=series_type)

            im = io.imread(str(im_paths[0]))
            x = np.array(len(im_paths), *im.shape)
            x[0] = im

            for i in range(1, len(im_paths)):
                x[i] = io.imread(str(im_paths[i]))
            res.append(x)
        return res

    def __len__(self):
        return len(self._patient_data) if not self.debug else 3


def get_datasets(seed,
                 patient_meta: pd.DataFrame,
                 debug=False,
                 train_val_split=False,
                 test=False,
                 fold_number=0,
                 normalisation="minmax",
                 limit_to_imaging_plane: Optional[str] = None,
                 limit_to_series_types: Optional[Tuple[str]] = None):
    if train_val_split and test:
        raise ValueError('Set either train_val_split to return train and '
                         'validation sets or test to return a test set')

    dataset_type = 'test' if test else 'train'
    base_folder = get_base_folder(
        dataset_type=dataset_type)
    patient_dirs = get_patient_dirs(base_folder=base_folder)

    def _filter_to_imaging_plane(imaging_plane: str, patient_dirs: List[Path]):
        if imaging_plane not in IMAGING_PLANES:
            raise ValueError(f'Imaging plane must be in {IMAGING_PLANES}')
        filtered = patient_meta[
            patient_meta['imaging_plane'] == limit_to_imaging_plane]
        patient_ids = filtered.index.unique()
        patient_dirs = [x for x in patient_dirs if x.name in patient_ids]
        return patient_dirs

    if limit_to_imaging_plane:
        patient_dirs = _filter_to_imaging_plane(
            imaging_plane=limit_to_imaging_plane, patient_dirs=patient_dirs)

    if test:
        return Brats(patient_dirs=patient_dirs, normalisation=normalisation,
                     dataset_type='test',
                     limit_to_series_types=limit_to_series_types)

    if train_val_split:
        kfold = KFold(5, shuffle=True, random_state=seed)
        splits = list(kfold.split(patient_dirs))
        train_idx, val_idx = splits[fold_number]
        print("first idx of train", train_idx[0])
        print("first idx of test", val_idx[0])
        train_patient_dirs = [patient_dirs[i] for i in train_idx]
        val_patient_dirs = [patient_dirs[i] for i in val_idx]
        train_dataset = Brats(patient_dirs=train_patient_dirs,
                              dataset_type='train',
                              debug=debug,
                              normalisation=normalisation,
                              limit_to_series_types=limit_to_series_types,
                              meta=patient_meta)
        val_dataset = Brats(patient_dirs=val_patient_dirs, dataset_type='val',
                            debug=debug,
                            normalisation=normalisation,
                            meta=patient_meta)
        bench_dataset = Brats(patient_dirs=val_patient_dirs,
                              dataset_type='val',
                              debug=debug,
                              normalisation=normalisation,
                              limit_to_series_types=limit_to_series_types,
                              meta=patient_meta)
        return train_dataset, val_dataset, bench_dataset
    else:
        train_dataset = Brats(patient_dirs=patient_dirs, debug=debug,
                              dataset_type='train',
                              normalisation=normalisation,
                              limit_to_series_types=limit_to_series_types,
                              meta=patient_meta)
        bench_dataset = Brats(patient_dirs=patient_dirs, dataset_type='train',
                              debug=debug,
                              normalisation=normalisation,
                              limit_to_series_types=limit_to_series_types,
                              meta=patient_meta)
        return train_dataset, bench_dataset
