import json
import math
import random
from pathlib import Path

import tensorstore
import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([1.6399796189744236, 1.790703594474254, -0.45981063270182365, 0.36037534690736855, -0.11991601069819327, -0.05995317416806099, -0.18829815529791255])
STD = torch.tensor([5.781253985376394, 5.003941058357719, 6.096484699468417, 0.22573912799066284, 0.46552001098903634, 0.5430271823494489, 0.5041364764608589])

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

FEATURES = [f'acc_{x}' for x in ('x', 'y', 'z')] + [f'rot_{x}' for x in ('w', 'x', 'y', 'z')]

class BFRBDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        meta_path: Path,
        is_train: bool,
        window_length: int = 64,
        features: list[str] = FEATURES
    ):
        """

        :param data_path:
        :param meta_path:
        :param window_length: sequence length to extract from timeseries. 99.9% of examples
            have a gesture length under 64 in the train set
        """
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

        feature_idxs = [FEATURES.index(x) for x in self._features]
        x = torch.tensor(self._data[arr_idx, start:end, feature_idxs].read().result(), dtype=torch.float)  # (T, C)
        x = (x - MEAN[feature_idxs]) / STD[feature_idxs]
        y = torch.tensor(actions[start:end], dtype=torch.long)

        sequence_label = gesture
        return x, y, sequence_label

    @property
    def num_channels(self):
        return len(self._features)
