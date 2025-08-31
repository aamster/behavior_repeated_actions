import json
import math
import random
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
import polars

MEAN = {'acc_x': 0.2859063536132305, 'acc_y': 3.484702360852425, 'acc_z': -3.3385503986300584, 'rot_x': -0.06656532651801965, 'rot_y': 0.003474407312620778, 'rot_z': -0.037349534369027416, 'rot_w': 0.2922282638187124, 'acc_mag': 10.067906067352848, 'rot_angle': 2.5332424289635345, 'acc_mag_jerk': -0.00023848618109139843, 'rot_angle_vel': -0.0013106331163317126, 'linear_acceleration_x': -0.03330287431651595, 'linear_acceleration_y': -0.03204492536032165, 'linear_acceleration_z': -0.11820445279716499, 'linear_acc_mag': 1.789874433776744, 'linear_acc_mag_jerk': 0.08765613004759497, 'angular_velocity_x': -0.42442241496677546, 'angular_velocity_y': 0.8689287849646535, 'angular_velocity_z': 0.8005171000441262, 'angular_distance': 0.11942487538993812}
STD = {'acc_x': 5.966392904390007, 'acc_y': 4.8711757864544465, 'acc_z': 4.569688674039315, 'rot_x': 0.5029527799524269, 'rot_y': 0.6377548690044493, 'rot_z': 0.4630912189426574, 'rot_w': 0.18605990792421245, 'acc_mag': 1.4887348923358488, 'rot_angle': 0.4095682211995221, 'acc_mag_jerk': 2.115899583377374, 'rot_angle_vel': 0.11322483956641816, 'linear_acceleration_x': 2.000550551168048, 'linear_acceleration_y': 1.3241942399113404, 'linear_acceleration_z': 1.6795665035757747, 'linear_acc_mag': 2.32144228775549, 'linear_acc_mag_jerk': 1.7855384746534821, 'angular_velocity_x': 18.704371115317596, 'angular_velocity_y': 28.23899987096596, 'angular_velocity_z': 21.78382309371009, 'angular_distance': 0.1622431451464932}

class BehaviorType(Enum):
    BFRB = 'BFRB'
    NON_BFRB = 'NON_BFRB'
    SETUP = 'SETUP'

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

SETUP_BEHAVIORS = {k: v for k, v in ACTION_ID_MAP.items() if 0 <= v <= 2}
BFRB_BEHAVIORS = {k: v for k, v in ACTION_ID_MAP.items() if 3 <= v <= 10}
NON_BFRB_BEHAVIORS = {k: v for k, v in ACTION_ID_MAP.items() if 11 <= v <= 20}


PAD_TOKEN_ID = len(ACTION_ID_MAP) + 1

RAW_FEATURES = [
    'acc_x',
    'acc_y',
    'acc_z',
    'rot_w',
    'rot_y',
    'rot_x',
    'rot_z',
]

ORIENTATION_MAP = {
    'Seated Lean Non Dom - FACE DOWN': 0,
    'Lie on Side - Non Dominant': 1,
    'Seated Straight': 2,
    'Lie on Back': 3
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
        with open(meta_path) as f:
            meta = json.load(f)

        self._meta = meta
        self._window_length = window_length
        self._is_train = is_train
        self._features = features
        self._data_path = data_path

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, idx, gesture_fraction: float = 0.2, sample_setup_window: bool = False):
        meta = self._meta[idx]
        gesture_start = meta["gesture_start"]
        actions = meta["actions"]
        sequence_length = meta["sequence_length"]
        handedness = meta["handedness"]
        orientation = ORIENTATION_MAP[meta["orientation"]]
        gesture = actions[gesture_start]
        sequence_id = meta["sequence_id"]

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

        x = polars.read_parquet(
            source=self._data_path / f'sequence_id={sequence_id}',
        ).drop('sequence_id')
        x = x[start:end]
        if self._features is not None:
            x = x[[c for c in self._features if c not in ("handedness", "orientation")]]

        x =  x.to_torch().float()

        if self._features is not None:
            mean = {k: v for k, v in MEAN.items() if k in self._features}
            std = {k: v for k, v in STD.items() if k in self._features}
        else:
            mean = MEAN
            std = STD

        x = (x - torch.tensor(list(mean.values()))) / torch.tensor(list(std.values()))

        y = torch.tensor(actions[start:end], dtype=torch.long)

        sequence_label = gesture - min(BFRB_BEHAVIORS.values())

        if self._features is not None and "handedness" not in self._features:
            handedness = None
        if self._features is not None and "orientation" not in self._features:
            orientation = None
        return x, y, sequence_label, handedness, orientation

    @property
    def num_channels(self):
        if self._features is None:
            num_channels = len(pd.read_parquet(path=self._data_path / f'sequence_id={self._meta[0]["sequence_id"]}').drop(columns=['sequence_id']).columns)
        else:
            num_channels = len(self._features)
            if "handedness" in self._features:
                num_channels -= 1
            if "orientation" in self._features:
                num_channels -= 1
        return num_channels