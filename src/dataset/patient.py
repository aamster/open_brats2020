from pathlib import Path
from typing import Optional


class Patient:
    def __init__(self,
                 id: str,
                 T1w_path: Path,
                 T1wCE_path: Path,
                 FLAIR_path: Path,
                 T2w: Path,
                 mgmt_val: Optional[int] = None):
        self._id = id
        self._t1_path = T1w_path
        self._t1ce_path = T1wCE_path
        self._flair_path = FLAIR_path
        self._t2_path = T2w
        self._mgmt_val = mgmt_val

    @property
    def id(self):
        return self._id

    @property
    def T1w_path(self):
        return self._t1_path

    @property
    def T1wCE_path(self):
        return self._t1ce_path

    @property
    def FLAIR_path(self):
        return self._flair_path

    @property
    def mgmt_val(self):
        return self._mgmt_val
