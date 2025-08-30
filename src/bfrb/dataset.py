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

MEAN = {'acc_x': 0.2859063536132305, 'acc_y': 3.484702360852425, 'acc_z': -3.3385503986300584, 'rot_w': 0.2922282638187124, 'rot_x': -0.06656532651801965, 'rot_y': 0.003474407312620778, 'rot_z': -0.037349534369027416, 'x_earth_x': -0.24518373397585927, 'x_earth_y': 0.08208037501189866, 'x_earth_z': 0.03253916696531566, 'y_earth_x': 0.13469615087827685, 'y_earth_y': 0.05351810735041136, 'y_earth_z': 0.35848596189732457, 'ang_velocity_x': -0.014854784523837166, 'ang_velocity_y': 0.030412507473762958, 'ang_velocity_z': 0.02801809850154442, 'a_vert': 9.746300681775748, 'a_horz': 1.4587260212655349, 'w_spin': -0.00173223223204785, 'w_tilt': 0.6598662813986627, 'tilt': 1.9531988245090337, 'theta_rate': 0.8359741277295686, 'jerk_rate': 16.582707009879048, 'acc_earth_x': 0.02360238732026308, 'acc_earth_y': -0.053997061177420015, 'theta_spin_rate': -0.0013186247823754227, 'theta_tilt_rate': 0.6558297634725405, 'acc_vertical_rate': -0.29200118675546094, 'acc_horizontal_rate': 0.5069349183234549, 'acc_dd_mag_rate': 192.4702400821918, 'acc_horizontal_std_w': 0.7117620596490517, 'acc_horizontal_over_L': 5.802972452977738, 'acc_vertical_over_L': 38.79373311802501, 'jerk_rate_over_L': 65.85741480188229, 'a_cen_over_g': 0.05102344765721896, 'a_tan_over_g': 0.20162751079753205, 'v_nd': 0.13398820729530705}

STD = {'acc_x': 5.966392904390007, 'acc_y': 4.8711757864544465, 'acc_z': 4.569688674039315, 'rot_w': 0.18605990792421245, 'rot_x': 0.5029527799524269, 'rot_y': 0.6377548690044493, 'rot_z': 0.4630912189426574, 'x_earth_x': 0.5402969844266604, 'x_earth_y': 0.5531655290584637, 'x_earth_z': 0.5780795942823888, 'y_earth_x': 0.6068914886777793, 'y_earth_y': 0.49870258477627877, 'y_earth_z': 0.4831759794885121, 'ang_velocity_x': 0.6546529890361168, 'ang_velocity_y': 0.9883649954838085, 'ang_velocity_z': 0.7624338082798533, 'a_vert': 1.601348660919818, 'a_horz': 1.9739523910100338, 'w_spin': 0.7839600776560961, 'w_tilt': 0.9688404915786197, 'tilt': 0.5223365853541633, 'theta_rate': 1.13570201602545, 'jerk_rate': 21.372858179112406, 'acc_earth_x': 1.6902562143090505, 'acc_earth_y': 1.7787441698507547, 'theta_spin_rate': 0.8057037979291475, 'theta_tilt_rate': 0.9536240265493907, 'acc_vertical_rate': 15.879713275979476, 'acc_horizontal_rate': 11.606454876642841, 'acc_dd_mag_rate': 258.8622462978494, 'acc_horizontal_std_w': 0.846395581063555, 'acc_horizontal_over_L': 7.859103235836185, 'acc_vertical_over_L': 7.187176559398683, 'jerk_rate_over_L': 84.76246794256257, 'a_cen_over_g': 0.18317336537501372, 'a_tan_over_g': 0.2805227972300836, 'v_nd': 0.18185325942365946}

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
            num_channels = len(pd.read_parquet(path=self._data_path / f'sequence_id={self._meta[0]["sequence_id"]}').drop('sequence_id').columns)
        else:
            num_channels = len(self._features)
            if "handedness" in self._features:
                num_channels -= 1
            if "orientation" in self._features:
                num_channels -= 1
        return num_channels