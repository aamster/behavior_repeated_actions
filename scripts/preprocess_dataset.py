import json
from pathlib import Path
from typing import Optional

import click
import loguru
import numpy as np
import pandas as pd
import tensorstore
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from bfrb.dataset import ACTION_ID_MAP, CHANNEL_IDX_MAP, RAW_CHANNELS


def _calculate_angular_velocity(
        rotation: Rotation,
        acceleration: np.ndarray,
        sampling_rate: int = 7    # assuming 7 Hz sampling rate
):
    T = rotation.as_matrix().shape[0]

    dt = np.array([1/sampling_rate] * (T-1))

    # Relative rotation from t-1 -> t
    rel = rotation[:-1].inv() * rotation[1:]            # [T-1] Rotation objects
    rotvec = rel.as_rotvec()                  # [T-1,3] axis*angle (radians)
    omega = np.zeros_like(acceleration)           # [T,3]
    omega[1:] = rotvec / dt[:, None]          # rad/s (approx)

    return omega

def extract_features(orientation: np.ndarray, linear_acceleration: np.ndarray, quat_order: str = "wxyz"):
    assert orientation.shape[-1] == 4 and linear_acceleration.shape[-1] == 3
    assert orientation.shape[0] == linear_acceleration.shape[0]

    rot = Rotation.from_quat(
        orientation,
        scalar_first=(quat_order.lower() == "wxyz")
    )

    R_earth = rot.as_matrix()                        # [T,3,3]
    x_earth = R_earth[:, :, 0]                       # [T,3]
    y_earth = R_earth[:, :, 1]                       # [T,3]

    # x, y should be unit and orthogonal
    assert np.allclose(np.linalg.norm(x_earth, axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(y_earth, axis=1), 1, atol=1e-5)
    assert np.allclose((x_earth * y_earth).sum(1), 0, atol=1e-4)

    ang_velocity = _calculate_angular_velocity(
        rotation=rot,
        acceleration=linear_acceleration,

    )
    feat = np.concatenate([x_earth, y_earth, ang_velocity], axis=1)
    return feat

def get_excluded_sequences(data: pd.DataFrame):
    data = data.copy()
    data = data.set_index('sequence_id')
    bad = set()

    for seq_id in data.index.unique():
        if np.isnan(data.loc[seq_id][RAW_CHANNELS]).any().any():
            bad.add(seq_id)
        elif not 'Performs gesture' in data.loc[seq_id]['behavior'].unique():
            bad.add(seq_id)
    return bad

def open_tensorstore(
    path: str,
    create: bool,
    chunk_shape: Optional[tuple[int, ...]] = None,
    shard_shape: Optional[tuple[int, ...]] = None,
    overwrite: bool = False,
    dtype: Optional[tensorstore.dtype] = None,
    shape: Optional[tuple[int, ...]] = None,
    read: bool = False
) -> tensorstore.TensorStore:
    """
    Parameters
    ----------
    path: Base path to pyramid containing all levels.
    create: Whether to create
    chunk_shape: tensorstore chunk shape
    shard_shape: tensorstore chunk shape
    overwrite: Whether to delete the existing tensorstore before writing
    dtype: tensorstore datatype
    shape: tensorstore shape

    Returns
    -------
    The opened tensorstore.Tensorstore
    """
    kvstore = {
        'driver': 'file',
        'path': path
    }

    if create:
        if chunk_shape is None or shard_shape is None:
            raise ValueError("Must provide chunk_shape and shard_shape if creating")
    compression_codec = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": "shuffle",
        },
    }

    kwargs = {}
    if chunk_shape is not None and shard_shape is not None:
        kwargs["chunk_layout"] = tensorstore.ChunkLayout(
            read_chunk=tensorstore.ChunkLayout.Grid(shape=chunk_shape),
            write_chunk=tensorstore.ChunkLayout.Grid(shape=shard_shape),
        )

    arr = tensorstore.open(
        spec={
            "driver": "zarr3",
            "kvstore": kvstore,
        },
        create=create,
        delete_existing=overwrite,
        dtype=dtype,
        shape=shape,
        codec=tensorstore.CodecSpec({"driver": "zarr3", "codecs": [compression_codec]}),
        read=read,
        **kwargs,
    ).result()

    return arr

def write_data(input_data_path: Path, subject_meta_path: Path, out_path: Path):
    data = pd.read_csv(input_data_path)
    excluded = get_excluded_sequences(data=data)
    data = data[~data['sequence_id'].isin(excluded)]
    data = data.set_index('sequence_id')

    subject_meta = pd.read_csv(subject_meta_path)
    subject_meta = subject_meta.set_index('subject')

    max_seq_length = data.groupby('sequence_id').size().max()

    shape = (data.index.nunique(), max_seq_length, len(CHANNEL_IDX_MAP))

    preprocessed = open_tensorstore(
        path=str(out_path),
        chunk_shape=(1, max_seq_length, shape[2]),
        shard_shape=shape,
        overwrite=True,
        dtype=tensorstore.float64,
        shape=shape,
        create=True,
        read=False
    )

    sequence_ids = data.index.unique()
    metadata = []
    for sequence_idx, sequence_id in tqdm(enumerate(sequence_ids), total=len(sequence_ids)):
        sequence_length = len(data.loc[sequence_id])
        subject_id = data.loc[sequence_id]['subject'].iloc[0]

        for col_name, channel_idx in {k: v for k, v in CHANNEL_IDX_MAP.items() if k in RAW_CHANNELS}.items():
            values = data.loc[sequence_id][col_name].values
            preprocessed[sequence_idx, :sequence_length, channel_idx] = values

        acc_idxs = [CHANNEL_IDX_MAP[f'acc_{x}'] for x in ('x', 'y', 'z')]
        orientation_idxs = [CHANNEL_IDX_MAP[f'rot_{x}'] for x in ('w','x','y','z')]
        pose_x_earth_coords_idxs = [CHANNEL_IDX_MAP[f'pose_x{x}_earth_coords'] for x in ('x', 'y', 'z')]
        pose_y_earth_coords_idxs = [CHANNEL_IDX_MAP[f'pose_y{x}_earth_coords'] for x in
                                    ('x', 'y', 'z')]
        angular_velocity_idxs = [CHANNEL_IDX_MAP[f'angular_velocity_{x}'] for x in ('x','y','z')]

        raw = preprocessed[sequence_idx, :sequence_length].read().result()

        features = extract_features(
            orientation=raw[:, orientation_idxs],
            linear_acceleration=raw[:, acc_idxs],
        )

        preprocessed[sequence_idx, :sequence_length, pose_x_earth_coords_idxs] = features[:, :3]
        preprocessed[sequence_idx, :sequence_length, pose_y_earth_coords_idxs] = features[:, 3:6]
        preprocessed[sequence_idx, :sequence_length, angular_velocity_idxs] = features[:, 6:]

        actions = []
        gesture = data.loc[sequence_id]['gesture'].iloc[0]
        for i, behavior in enumerate(data.loc[sequence_id]['behavior']):
            if behavior == 'Performs gesture':
                actions.append(ACTION_ID_MAP[gesture.lower()])
            else:
                actions.append(ACTION_ID_MAP[behavior.lower()])

        gesture_start = np.where((data.loc[sequence_id]['behavior'] == 'Performs gesture').values)[0][0]

        meta = {
            'sequence_id': sequence_id,
            'arr_idx': sequence_idx,
            'actions': actions,
            'gesture_start': int(gesture_start),
            'sequence_length': len(data.loc[sequence_id]),
            'handedness': subject_meta.loc[subject_id]['handedness'],
            "subject_id": subject_id,
        }
        metadata.append(meta)

    with open(out_path / "meta.json", 'w') as f:
        f.write(json.dumps(metadata, indent=2))

class DataSplitter:
    def __init__(self, n_examples: int, train_frac: float, rng=None):
        self._n_examples = n_examples
        self._train_frac = train_frac
        self._rng = rng

    def split(self):
        idxs = np.arange(self._n_examples)

        if self._rng is not None:
            self._rng.shuffle(idxs)
        else:
            np.random.shuffle(idxs)

        n_train = int(self._n_examples * self._train_frac)

        train_idxs = idxs[:n_train]
        test_idxs = idxs[n_train:]

        return train_idxs, test_idxs

def write_train_test_split(input_path: Path, out_path: Path, train_frac: float = 0.8):
    data = open_tensorstore(
        path=str(input_path),
        create=False,
        read=True
    )

    with open(input_path / 'meta.json') as f:
        meta = json.load(f)

    subjects = sorted(list(set([x['subject_id'] for x in meta])))

    rng = np.random.default_rng(seed=1234)
    splitter = DataSplitter(n_examples=len(subjects), train_frac=train_frac, rng=rng)
    train_suject_idxs, val_subject_idxs = splitter.split()

    train_subjects = [subjects[i] for i in train_suject_idxs]
    val_subjects = [subjects[i] for i in val_subject_idxs]

    train_meta = [x for x in meta if x['subject_id'] in train_subjects]
    val_meta = [x for x in meta if x['subject_id'] in val_subjects]

    train_idxs = [x['arr_idx'] for x in train_meta]
    val_idxs = [x['arr_idx'] for x in val_meta]

    loguru.logger.info(f'n train: {len(train_idxs)}')
    loguru.logger.info(f'n val: {len(val_idxs)}')

    train = open_tensorstore(
        path=str(out_path / 'train.zarr'),
        create=True,
        chunk_shape=data.chunk_layout.read_chunk.shape,
        shard_shape=data.chunk_layout.write_chunk.shape,
        dtype=data.dtype,
        shape=(len(train_idxs), *data.shape[1:]),
        overwrite=True
    )

    val = open_tensorstore(
        path=str(out_path / 'val.zarr'),
        create=True,
        chunk_shape=data.chunk_layout.read_chunk.shape,
        shard_shape=data.chunk_layout.write_chunk.shape,
        dtype=data.dtype,
        shape=(len(val_idxs), *data.shape[1:]),
        overwrite=True
    )

    train[:] = data[train_idxs]
    val[:] = data[val_idxs]

    for i, x in enumerate(train_meta):
        x['arr_idx'] = i

    for i, x in enumerate(val_meta):
        x['arr_idx'] = i

    with open(out_path / 'train.json', 'w') as f:
        f.write(json.dumps(train_meta, indent=2))

    with open(out_path / 'val.json', 'w') as f:
        f.write(json.dumps(val_meta, indent=2))

def get_stats(input_path: Path):
    train = open_tensorstore(
        path=str(input_path / 'train.zarr'),
        create=False,
        read=True
    )

    with open(input_path / 'train.json') as f:
        meta = json.load(f)

    x = train[:].read().result()

    C = x.shape[-1]
    sum_ = np.zeros(C, dtype=np.float64)
    sumsq_ = np.zeros(C, dtype=np.float64)
    N = 0

    for row_idx in range(x.shape[0]):
        seq_length = meta[row_idx]['sequence_length']
        gesture_start = meta[row_idx]['gesture_start']
        sum_ += np.sum(x[row_idx, gesture_start:seq_length], axis=0)
        sumsq_ += np.square(x[row_idx, gesture_start:seq_length]).sum(axis=0)
        N += seq_length

    mean = sum_ / N

    # Var = E[x^2] - (E[x])^2
    var = sumsq_ / N - np.square(mean)
    # Clamp tiny negatives due to round-off
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    return mean.tolist(), std.tolist()

@click.command()
@click.option(
    '--input-data-dir',
    type=click.Path(path_type=Path, readable=True, dir_okay=True)
)
@click.option(
    '--out-path',
    type=click.Path(path_type=Path, writable=True)
)
def main(input_data_dir: Path, out_path: Path):
    # write_data(
    #     input_data_path=input_data_dir / 'train.csv',
    #     out_path=out_path,
    #     subject_meta_path=input_data_dir / 'train_demographics.csv',
    # )
    # write_train_test_split(
    #     input_path=out_path,
    #     out_path=out_path.parent,
    # )
    mean, std = get_stats(input_path=out_path.parent)
    print(mean)
    print(std)

if __name__ == '__main__':
    main()
