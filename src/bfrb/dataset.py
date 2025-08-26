import json
import math
import random
from enum import Enum
from pathlib import Path
from typing import Optional

import tensorstore
import torch
from torch.utils.data import Dataset

MEAN = torch.tensor(
    [0.1259477966683311, 1.535085103383252, -1.4707020724081459, 0.1287327318439677,
     -0.02932343441655031, 0.0015305499168615695, -0.016453259960534938, -0.10800862129474795,
     0.03615814147468665, 0.014334191363401746, 0.05933650374826441, 0.02357585837970549,
     0.15792064942556056, -0.006543846812481386, 0.013397352871217698, 0.0123425650689789,
     4.293451618068218, 0.6425996694347449, -0.0007630849408804302, 0.2906850553951474,
     0.8604253990622166, 0.3682642869899371, 7.305033219080255, 0.010397350886161375,
     -0.02378684767202186, -0.0005808821100810843, 0.28890688386246766, -0.12863269959414533,
     0.22331555493674402, 84.78721216307726, 0.31354624348845855, 0.0, 0.0]

)
STD = torch.tensor(
    [3.962543356743154, 3.666833795106159, 3.4563036864505814, 0.19051854299253687,
     0.33545037104476966, 0.4232926938469801, 0.3079206146210989, 0.3786996120256004,
     0.3694001307113988, 0.38402155956006884, 0.40831743848730173, 0.3320623514531971,
     0.3667656531355792, 0.4345673392862545, 0.6561690445779216, 0.5062319259347158,
     4.953905505097491, 1.49697292243475, 0.5203288714680956, 0.721672697257727, 1.029776589136225,
     0.8604844072425396, 16.40132211159362, 1.1219141169827798, 1.1808882507450125,
     0.5347602529923056, 0.7117695884429263, 10.540643271583296, 7.707519108084458,
     196.59429690634062, 0.6636580678724984, 0.0, 0.0]

)

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

CHANNEL_IDX_MAP = {
    "acc_x": 0, "acc_y": 1, "acc_z": 2,
    "rot_w": 3, "rot_x": 4, "rot_y": 5, "rot_z": 6,
    "pose_xx_earth_coords": 7, "pose_xy_earth_coords": 8, "pose_xz_earth_coords": 9,
    "pose_yx_earth_coords": 10, "pose_yy_earth_coords": 11, "pose_yz_earth_coords": 12,
    "angular_velocity_x": 13, "angular_velocity_y": 14, "angular_velocity_z": 15,
    "acc_vertical": 16, "acc_horizontal": 17,
    "w_spin": 18, "w_tilt": 19,
    "tilt": 20,
    "theta_rate": 21,
    "jerk_rate": 22,
    "acc_earth_x": 23, "acc_earth_y": 24,
    "theta_spin_rate": 25, "theta_tilt_rate": 26,
    "acc_vertical_rate": 27, "acc_horizontal_rate": 28,
    "acc_dd_mag_rate": 29, "acc_horizontal_std_w": 30,
    "handedness": 31, "orientation": 32
}

RAW_CHANNELS = [x for x in CHANNEL_IDX_MAP if x in ('acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z')]

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
        handedness = meta["handedness"]
        orientation = ORIENTATION_MAP[meta["orientation"]]
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

        feature_idxs = [CHANNEL_IDX_MAP[x] for x in self._features if x not in ("handedness", "orientation")]
        x =  torch.tensor(self._data[arr_idx, start:end, feature_idxs].read().result(), dtype=torch.float)  # (T, C)
        x = (x - MEAN[feature_idxs]) / STD[feature_idxs]

        y = torch.tensor(actions[start:end], dtype=torch.long)


        return x, y, gesture, handedness if "handedness" in self._features else None, orientation if "orientation" in self._features else None

    @property
    def num_channels(self):
        num_channels = len(self._features)
        if "handedness" in self._features:
            num_channels -= 1 # it gets added separately
        if "orientation" in self._features:
            num_channels -= 1
        return num_channels

    @property
    def num_raw_classes(self) -> int:
        return len(ACTION_ID_MAP)

    @property
    def num_sequence_classes(self) -> int:
        """
        The dataset contains labels for 3 setup + 8 BFRB + 10 non-BFRB
        but task is only to predict sequence labels for 8 BFRB + non-BFRB

        :return:
        """
        return len(BFRB_BEHAVIORS) + 1