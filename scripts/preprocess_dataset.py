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


def extract_features(
    orientation: np.ndarray,
    linear_acceleration: np.ndarray,    # [T,3] in device frame
    quat_order: str = "wxyz",
    dt_sec: float = 1/7                 # sampling-time estimate
):
    """
    Returns features in this order (new ones marked '*'):

      pose_xx_earth_coords, pose_xy_earth_coords, pose_xz_earth_coords,
      pose_yx_earth_coords, pose_yy_earth_coords, pose_yz_earth_coords,
      angular_velocity_x, angular_velocity_y, angular_velocity_z,
      acc_vertical, acc_horizontal, w_spin, w_tilt, tilt,
      theta_rate, jerk_rate,
      *acc_earth_x, *acc_earth_y,
      *theta_spin_rate, *theta_tilt_rate,
      *acc_vertical_rate, *acc_horizontal_rate
    """
    assert orientation.shape[-1] == 4 and linear_acceleration.shape[-1] == 3
    assert orientation.shape[0] == linear_acceleration.shape[0]
    T = orientation.shape[0]

    # normalize quats (defensive)
    q = orientation / np.linalg.norm(orientation, axis=1, keepdims=True)
    rot = Rotation.from_quat(q, scalar_first=(quat_order.lower() == "wxyz"))

    # Rotation: device -> earth. Columns are device axes in earth coords
    R_earth = rot.as_matrix()                 # [T,3,3]
    x_earth = R_earth[:, :, 0]                # [T,3]
    y_earth = R_earth[:, :, 1]                # [T,3]
    # (z_earth would be R_earth[:,:,2] = x×y; we omit to avoid redundancy)

    # gravity direction in device frame: g_hat = R^T * e_z
    ez = np.array([0.0, 0.0, 1.0])
    g_hat = np.matmul(R_earth.transpose(0, 2, 1), ez)     # [T,3]
    g_hat = g_hat / np.linalg.norm(g_hat, axis=1, keepdims=True)

    # Acceleration splits
    a_dev = linear_acceleration
    a_vert = (a_dev * g_hat).sum(axis=1)                  # scalar vertical
    a_hvec = a_dev - a_vert[:, None] * g_hat
    a_horz = np.linalg.norm(a_hvec, axis=1)               # scalar horizontal magnitude

    # Earth-frame acceleration (new horizontal components)
    a_earth = np.matmul(R_earth, a_dev[:, :, None]).squeeze(-1)  # [T,3]
    acc_earth_x = a_earth[:, 0]
    acc_earth_y = a_earth[:, 1]
    # (a_earth[:,2] == a_vert)

    # Angular velocity in device frame (keep as-is)
    ang_velocity = _calculate_angular_velocity(rotation=rot, acceleration=linear_acceleration)  # [T,3]

    # Split gyro about gravity
    w_spin = (ang_velocity * g_hat).sum(axis=1)                  # twist about g (signed)
    w_tilt = np.linalg.norm(ang_velocity - w_spin[:, None] * g_hat, axis=1)

    # Tilt (angle between device z-axis and gravity) in radians
    tilt = np.arccos(np.clip(g_hat[:, 2], -1.0, 1.0))

    # Quaternion geodesic change (total) → rate
    delta = rot[:-1].inv() * rot[1:]
    rotvec = delta.as_rotvec()                                   # [T-1,3]
    theta_step = np.linalg.norm(rotvec, axis=1)                  # radians per step
    theta_rate = np.pad(theta_step / dt_sec, (1, 0))             # align to [T]

    # * Split quaternion change into spin vs tilt rates (new)
    eps = 1e-12
    axis = np.zeros_like(rotvec)
    nz = theta_step > eps
    axis[nz] = rotvec[nz] / theta_step[nz, None]                 # unit rotation axis per step

    g_mid = (g_hat[:-1] + g_hat[1:])                             # mid-step gravity dir
    g_mid_norm = np.linalg.norm(g_mid, axis=1, keepdims=True)
    valid = g_mid_norm.squeeze(-1) > eps
    g_mid[valid] = g_mid[valid] / g_mid_norm[valid]

    spin_frac_signed = np.zeros(T-1)
    if np.any(valid):
        spin_frac_signed[valid] = np.sum(axis[valid] * g_mid[valid], axis=1)   # ∈[-1,1]

    theta_spin_rate = np.pad((theta_step * spin_frac_signed) / dt_sec, (1, 0))        # signed
    theta_tilt_rate = np.pad(theta_step * np.sqrt(np.maximum(0.0, 1.0 - spin_frac_signed**2)) / dt_sec, (1, 0))

    # Jerk (linear accel diff) → rate
    jerk = np.linalg.norm(np.diff(a_dev, axis=0), axis=1)
    jerk_rate = np.pad(jerk / dt_sec, (1, 0))

    # * Component-wise accel change rates (new)
    acc_vertical_rate = np.pad(np.diff(a_vert), (1, 0)) / dt_sec
    acc_horizontal_rate = np.pad(np.diff(a_horz), (1, 0)) / dt_sec

    # Concatenate (drop per-step theta to avoid redundancy)
    feat = np.concatenate([
        x_earth,                      # 3
        y_earth,                      # 3
        ang_velocity,                 # 3
        a_vert[:, None],              # 1
        a_horz[:, None],              # 1
        w_spin[:, None],              # 1
        w_tilt[:, None],              # 1
        tilt[:, None],                # 1
        theta_rate[:, None],          # 1
        jerk_rate[:, None],           # 1
        acc_earth_x[:, None],         # 1  *
        acc_earth_y[:, None],         # 1  *
        theta_spin_rate[:, None],     # 1  *
        theta_tilt_rate[:, None],     # 1  *
        acc_vertical_rate[:, None],   # 1  *
        acc_horizontal_rate[:, None], # 1  *
    ], axis=1)

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

        raw = preprocessed[sequence_idx, :sequence_length].read().result()

        features = extract_features(
            orientation=raw[:, orientation_idxs],
            linear_acceleration=raw[:, acc_idxs],
        )

        new_feature_idxs = [i for k, i in CHANNEL_IDX_MAP.items() if k not in RAW_CHANNELS and k not in ("handedness", "orientation")]
        preprocessed[sequence_idx, :sequence_length, new_feature_idxs] = features

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
            "orientation": data.loc[sequence_id]['orientation'].iloc[0]
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
    write_data(
        input_data_path=input_data_dir / 'train.csv',
        out_path=out_path,
        subject_meta_path=input_data_dir / 'train_demographics.csv',
    )
    write_train_test_split(
        input_path=out_path,
        out_path=out_path.parent,
    )
    mean, std = get_stats(input_path=out_path.parent)
    print(mean)
    print(std)

if __name__ == '__main__':
    main()
