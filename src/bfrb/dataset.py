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
    [1.708104326580431, 1.8016810227020261, -0.4150403535361312, 0.3594622150221407,
     -0.12285486519490495, -0.05624846740477902, -0.1909408321459954, -0.1778527128302445,
     -0.07914598030655579, 0.16757626477402868, 0.2924598595066032, -0.04475735358540609,
     0.20365658033488682, 0.03898752130849018, -0.03947141240216166, -0.05098241052350764, 0.0]

)
STD = torch.tensor(
    [5.790281872199454, 4.975957261468285, 6.10009059329046, 0.22648961638963777,
     0.4641871113794403, 0.542168557596857, 0.5053288975408595, 0.5315202247021523,
     0.5732479875022234, 0.5682389615134157, 0.5794098947537498, 0.5334269698963092,
     0.5007273198182817, 0.8285049371837747, 1.034131242960337, 0.8845794994369625, 0.0]
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
    "handedness": 16, "orientation": 17
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