import json
import math
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import tensorstore
import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([1.6359909914071569, 1.8010360549022184, -0.4611939559243808, 2.1679728989417035, 1.8010360549022184, -0.4611939559243808, 0.3604065016258291, -0.12111841775745338, -0.06037032309477175, -0.18815398134691413])
STD = torch.tensor([5.781401244510568, 5.01774021718201, 6.0884138625190385, 5.603655983681371, 5.01774021718201, 6.0884138625190385, 0.22554675053571338, 0.46488208667061504, 0.5436761616628948, 0.5038059977764576])

ACTION_ID_MAP = {
    'moves hand to target location': 0,
    'relaxes and moves hand to target location': 1,
    'hand at target location': 2,
    'above ear - pull hair': 3,
    'forehead - pull hairline': 4,
    'forehead - scratch': 5,
    'eyebrow - pull hair': 6,
    'eyelash - pull hair': 7,
    'neck - pinch skin': 8,
    'neck - scratch': 9,
    'cheek - pinch skin': 10,
    'drink from bottle/cup': 11,
    'glasses on/off': 12,
    'pull air toward your face': 13,
    'pinch knee/leg skin': 14,
    'scratch knee/leg skin': 15,
    'write name on leg': 16,
    'text on phone': 17,
    'feel around in tray and pull out an object': 18,
    'write name in air': 19,
    'wave hello': 20
}

BFRB_BEHAVIORS = {k: v for k, v in ACTION_ID_MAP.items() if 3 <= v <= 10}
NON_BFRB_BEHAVIORS = {k: v for k, v in ACTION_ID_MAP.items() if 11 <= v <= 20}


PAD_TOKEN_ID = len(ACTION_ID_MAP) + 1

CHANNEL_IDX_MAP = {
    "acc_x": 0, "acc_y": 1, "acc_z": 2,
    "acc_x_mirror": 3, "acc_y_mirror": 4, "acc_z_mirror": 5,
    "rot_w": 6, "rot_x": 7, "rot_y": 8, "rot_z": 9,
}

class BFRBDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        meta_path: Path,
        is_train: bool,
        window_length: int = 64,
        features: Optional[list[str]] = None
    ):
        """

        :param data_path:
        :param meta_path:
        :param window_length: sequence length to extract from timeseries. 99.9% of examples
            have a gesture length under 64 in the train set
        """
        if features is None:
            features = list(CHANNEL_IDX_MAP.keys())

        data = tensorstore.open(spec={ # type: ignore
            'driver': 'zarr3',
            'kvstore': {
                'driver': 'file',
                'path': str(data_path)
            }
        }, read=True).result()

        with open(meta_path) as f:
            meta = json.load(f)

        self._data = data
        self._meta = meta
        self._window_length = window_length
        self._is_train = is_train
        self._features = features

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, idx, gesture_fraction: float = 0.2, sample_setup_window: bool = False):
        meta = self._meta[idx]
        arr_idx = meta["arr_idx"]
        gesture_start = meta["gesture_start"]
        actions = meta["actions"]
        sequence_length = meta["sequence_length"]
        gesture = actions[gesture_start]

        if self._is_train:
            if sample_setup_window and random.random() < 0.5 and gesture_start > self._window_length:
                # sample non-gesture
                max_start = max(0, gesture_start - self._window_length)
                start = random.randint(0, max_start)
                end = start + self._window_length
            else:
                # the gesture is always performed at the end of the sequence
                start = max(0, sequence_length - self._window_length)
                end = min(start + self._window_length, sequence_length)

                if sequence_length > self._window_length:
                    # choose a random portion that contains at least gesture_fraction gesture
                    gesture_length = sequence_length - gesture_start
                    max_shift = min(start, math.ceil(gesture_length * gesture_fraction))
                    shift = random.randint(0, max_shift)

                    start -= shift
                    end -= shift
        else:
            # the gesture is always performed at the end of the sequence
            start = max(0, sequence_length - self._window_length)
            end = min(start + self._window_length, sequence_length)

        feature_idxs = [CHANNEL_IDX_MAP[x] for x in self._features]
        x = torch.tensor(self._data[arr_idx, start:end, feature_idxs].read().result(), dtype=torch.float)  # (T, C)
        x = (x - MEAN[feature_idxs]) / STD[feature_idxs]
        y = torch.tensor(actions[start:end], dtype=torch.long)

        sequence_label = gesture
        return x, y, sequence_label

    @property
    def num_channels(self):
        return len(self._features)
