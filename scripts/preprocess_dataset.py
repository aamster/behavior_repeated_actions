import json
from pathlib import Path
from typing import Optional, Literal

import click
import loguru
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from bfrb.dataset import ACTION_ID_MAP, RAW_FEATURES


def _calculate_angular_velocity(
        rotation: Rotation,
        sampling_rate: int = 200
) -> np.ndarray:
    T = rotation.as_matrix().shape[0]

    dt = np.array([1/sampling_rate] * (T-1))

    # Relative rotation from t-1 -> t
    rel = rotation[:-1].inv() * rotation[1:]            # [T-1] Rotation objects
    rotvec = rel.as_rotvec()                  # [T-1,3] axis*angle (radians)
    omega = np.zeros((T, 3))           # [T,3]
    omega[1:] = rotvec / dt[:, None]          # rad/s (approx)

    return omega


def _calculate_angular_distance(
        rotation: Rotation,
) -> np.ndarray:
    delta_theta = np.zeros(rotation.as_matrix().shape[0])
    delta = rotation[:-1].inv() * rotation[1:]
    rotvec = delta.as_rotvec()
    delta_theta[1:] = np.linalg.norm(rotvec, axis=1)

    return delta_theta


def extract_features(rotation: np.ndarray, acceleration) -> pd.DataFrame:
    # Use the fast, simple pandas/Numpy path, then convert back to Polars.
    g = 9.81

    # --- base arrays ---
    acc_mag = np.linalg.norm(acceleration, axis=1)

    rot_w = np.clip(rotation[:, -1], -1.0, 1.0)
    rot_angle = 2 * np.arccos(rot_w)

    # per-sequence diffs (assumes your rows are already ordered within each sequence)
    acc_mag_jerk = np.pad(np.diff(acc_mag), (1, 0))
    rot_angle_vel = np.pad(np.diff(rot_angle), (1, 0))

    # gravity subtraction via SciPy (your fast path)
    rot = Rotation.from_quat(rotation)
    gravity = rot.apply(np.array([0.0, 0.0, g]), inverse=True)  # Nx3
    lin_acc = acceleration - gravity
    linear_acc_mag = np.linalg.norm(lin_acc, axis=1)
    linear_acc_mag_jerk = np.pad(np.diff(linear_acc_mag), (1, 0))

    # your fast implementations (unchanged)
    ang_vel = _calculate_angular_velocity(rot)      # shape [N, 3]
    ang_dist = _calculate_angular_distance(rot)     # shape [N]

    # final table (columns exactly as requested)
    out_pd = pd.DataFrame({
        "acc_mag": acc_mag,
        "rot_angle": rot_angle,
        "acc_mag_jerk": acc_mag_jerk,
        "rot_angle_vel": rot_angle_vel,
        "linear_acceleration_x": lin_acc[:, 0],
        "linear_acceleration_y": lin_acc[:, 1],
        "linear_acceleration_z": lin_acc[:, 2],
        "linear_acc_mag": linear_acc_mag,
        "linear_acc_mag_jerk": linear_acc_mag_jerk,
        "angular_velocity_x": ang_vel[:, 0],
        "angular_velocity_y": ang_vel[:, 1],
        "angular_velocity_z": ang_vel[:, 2],
        "angular_distance": ang_dist,
    })

    return out_pd

def get_excluded_sequences(data: pd.DataFrame):
    data = data.copy()
    data = data.set_index('sequence_id')
    bad = set()

    for seq_id in data.index.unique():
        if np.isnan(data.loc[seq_id][RAW_FEATURES]).any().any():
            bad.add(seq_id)
        elif not 'Performs gesture' in data.loc[seq_id]['behavior'].unique():
            bad.add(seq_id)
    return bad

def write_data(input_data_path: Path, subject_meta_path: Path, out_path: Path):
    data = pd.read_csv(input_data_path)
    excluded = get_excluded_sequences(data=data)
    data = data[~data['sequence_id'].isin(excluded)]
    data = data.set_index('sequence_id')

    subject_meta = pd.read_csv(subject_meta_path)
    subject_meta = subject_meta.set_index('subject')

    sequence_ids = data.index.unique()
    metadata = []
    acc_features = [f'acc_{x}' for x in ('x', 'y', 'z')]
    orientation_features = [f'rot_{x}' for x in ('x', 'y', 'z', 'w')]

    additional_features = []
    for sequence_idx, sequence_id in tqdm(enumerate(sequence_ids), total=len(sequence_ids)):
        subject_id = data.loc[sequence_id]['subject'].iloc[0]

        features = extract_features(
            rotation=data.loc[sequence_id, orientation_features].values,
            acceleration=data.loc[sequence_id, acc_features].values,
        )

        features['sequence_id'] = sequence_id
        additional_features.append(features)


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
            'height_cm': subject_meta.loc[subject_id]['height_cm'],
            'shoulder_to_wrist_cm': subject_meta.loc[subject_id]['shoulder_to_wrist_cm'],
            'forearm_cm': subject_meta.loc[subject_id]['elbow_to_wrist_cm'],
            'upper_arm_cm': subject_meta.loc[subject_id]['shoulder_to_wrist_cm'] - subject_meta.loc[subject_id]['elbow_to_wrist_cm'],
            "subject_id": subject_id,
            "orientation": data.loc[sequence_id]['orientation'].iloc[0]
        }
        metadata.append(meta)

    additional_features = pd.concat(additional_features).set_index('sequence_id')
    data = pd.concat([data, additional_features], axis=1).reset_index()

    data[['sequence_id'] + acc_features + orientation_features + list(additional_features.columns)].to_parquet(
        path=out_path,
        engine="pyarrow",
        index=False,
        partition_cols=["sequence_id"],
        compression="zstd"
    )

    with open(out_path.parent / "meta.json", 'w') as f:
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
    with open(input_path.parent / 'meta.json') as f:
        meta = json.load(f)

    subjects = sorted(list(set([x['subject_id'] for x in meta])))

    rng = np.random.default_rng(seed=1234)
    splitter = DataSplitter(n_examples=len(subjects), train_frac=train_frac, rng=rng)
    train_suject_idxs, val_subject_idxs = splitter.split()

    train_subjects = [subjects[i] for i in train_suject_idxs]
    val_subjects = [subjects[i] for i in val_subject_idxs]

    train_meta = [x for x in meta if x['subject_id'] in train_subjects]
    val_meta = [x for x in meta if x['subject_id'] in val_subjects]

    train_sequence_ids = [x['sequence_id'] for x in train_meta]
    val_sequence_ids = [x['sequence_id'] for x in val_meta]

    loguru.logger.info(f'n train: {len(train_sequence_ids)}')
    loguru.logger.info(f'n val: {len(val_sequence_ids)}')


    train = pl.scan_parquet(
        source=input_path,
    ).filter(pl.col("sequence_id").is_in(train_sequence_ids)).collect()

    val = pl.scan_parquet(
        source=input_path,
    ).filter(pl.col("sequence_id").is_in(val_sequence_ids)).collect()

    train.write_parquet(
        file=input_path.parent / 'train.parquet',
        partition_by=["sequence_id"],
    )

    val.write_parquet(
        file=input_path.parent / 'val.parquet',
        partition_by=["sequence_id"],
    )

    with open(out_path / 'train.json', 'w') as f:
        f.write(json.dumps(train_meta, indent=2))

    with open(out_path / 'val.json', 'w') as f:
        f.write(json.dumps(val_meta, indent=2))

def get_stats(input_path: Path):



    with open(input_path / 'train.json') as f:
        meta = json.load(f)


    C = pl.read_parquet(
        source=input_path / 'train.parquet' / f'sequence_id={meta[0]["sequence_id"]}',
    ).shape[-1] - 1
    sum_ = np.zeros(C, dtype=np.float64)
    sumsq_ = np.zeros(C, dtype=np.float64)
    N = 0

    sequence_ids = [x['sequence_id'] for x in meta]
    for i, seq_id in tqdm(enumerate(sequence_ids), total=len(sequence_ids)):
        seq = pl.read_parquet(
            source=input_path / 'train.parquet' / f'sequence_id={seq_id}',
        ).drop('sequence_id')

        seq_length = meta[i]['sequence_length']
        gesture_start = meta[i]['gesture_start']
        sum_ += seq[gesture_start:].sum().to_numpy()[0]
        sumsq_ += (seq[gesture_start:] * seq[gesture_start:]).sum().to_numpy()[0]
        N += (seq_length - gesture_start)

    mean = sum_ / N

    # Var = E[x^2] - (E[x])^2
    var = sumsq_ / N - np.square(mean)
    # Clamp tiny negatives due to round-off
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    column_names = pl.read_parquet(
        source=input_path / 'train.parquet' / f'sequence_id={meta[0]["sequence_id"]}',
    ).columns
    return {column_names[i]: mean[i] for i in range(len(mean))}, {column_names[i]: std[i] for i in range(len(std))}

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
