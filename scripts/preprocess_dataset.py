import json
from pathlib import Path
from typing import Optional, Literal

import click
import loguru
import numpy as np
import pandas as pd
import polars
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from bfrb.dataset import ACTION_ID_MAP, RAW_FEATURES


def _calculate_angular_velocity(
        rotation: Rotation,
        sampling_rate: int = 7    # assuming 7 Hz sampling rate
):
    T = rotation.as_matrix().shape[0]

    dt = np.array([1/sampling_rate] * (T-1))

    # Relative rotation from t-1 -> t
    rel = rotation[:-1].inv() * rotation[1:]            # [T-1] Rotation objects
    rotvec = rel.as_rotvec()                  # [T-1,3] axis*angle (radians)
    omega = np.zeros((T, 3))           # [T,3]
    omega[1:] = rotvec / dt[:, None]          # rad/s (approx)

    return omega


def append_size_norm_features(
    *,
    X: pd.DataFrame,                               # [T, F]
    dt_sec: float = 1/7,
    shoulder_to_wrist_cm: Optional[float] = None,
    forearm_cm: Optional[float] = None,
    which_radius: Literal["shoulder", "forearm"] = "shoulder",
) -> pd.DataFrame:
    """
    Appends a few length-normalized features using L and returns (X_out, cols_out).
    Uses device-frame mags for stability; leverages your existing columns when possible.
    """
    G = 9.8

    # ----- pick L (meters) -----
    L_cm_opt = forearm_cm if which_radius == "forearm" else shoulder_to_wrist_cm
    L: float = 0.01 * float(L_cm_opt)

    # ----- 1) simple normalized versions of existing features -----
    for base_name in ("acc_horizontal", "acc_vertical", "jerk_rate"):
        v = X[base_name] / max(L, 1e-6)
        new_name = f"{base_name}_over_L"
        X[new_name] = v

    # ----- 2) compact dimensionless kinematics (needs ω and α) -----
    w = X[[k for k in  ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z")]].values                # [T,3], rad/s
    omega = np.linalg.norm(w, axis=1)                         # [T]
    alpha = np.zeros_like(omega)                              # [T], rad/s^2
    if dt_sec > 0 and len(omega) > 1:
        dw = np.diff(w, axis=0) / dt_sec
        alpha[1:] = np.linalg.norm(dw, axis=1)

    a_cen_over_g = (omega ** 2) * L / G
    a_tan_over_g = alpha * L / G
    v_nd = omega * np.sqrt(L / G)

    for name, vec in (("a_cen_over_g", a_cen_over_g),
                      ("a_tan_over_g", a_tan_over_g),
                      ("v_nd", v_nd)):
        X[name] = vec

    return X

def extract_features(
    *,
    orientation: np.ndarray,                 # [T,4] quaternions
    linear_acceleration: np.ndarray,         # [T,3] in device frame
    forearm_cm: float,
    quat_order: Literal["wxyz", "xyzw"] = "wxyz",
    dt_sec: float = 1.0 / 7.0,
    horiz_std_window: int = 5,               # odd >=3
    eps: float = 1e-12,
    which_radius: Literal["shoulder", "forearm"] = "shoulder",
) -> pd.DataFrame:
    # Coerce window to odd >= 3
    if horiz_std_window < 3:
        horiz_std_window = 3
    if (horiz_std_window % 2) == 0:
        horiz_std_window += 1

    T: int = int(orientation.shape[0])
    dt: float = float(dt_sec)

    # Normalize quats and build rotation (device -> earth)
    q = orientation / (np.linalg.norm(orientation, axis=1, keepdims=True) + eps)
    rot = Rotation.from_quat(q, scalar_first=(quat_order.lower() == "wxyz"))
    R_earth = rot.as_matrix()                               # [T,3,3]
    x_earth = R_earth[:, :, 0]                              # [T,3]
    y_earth = R_earth[:, :, 1]                              # [T,3]

    # Gravity direction in device frame: g_hat = R^T * e_z
    ez = np.array([0.0, 0.0, 1.0])
    g_hat = (R_earth.transpose(0, 2, 1) @ ez)               # [T,3]
    g_hat /= (np.linalg.norm(g_hat, axis=1, keepdims=True) + eps)

    # Accel splits
    a_dev = linear_acceleration.astype(np.float64)          # [T,3]
    a_vert = np.einsum("td,td->t", a_dev, g_hat)            # [T]
    a_hvec = a_dev - a_vert[:, None] * g_hat                # [T,3]
    a_horz = np.linalg.norm(a_hvec, axis=1)                 # [T]

    # Earth-frame acceleration (x,y for horizontal)
    a_earth = (R_earth @ a_dev[:, :, None]).squeeze(-1)     # [T,3]
    acc_earth_x = a_earth[:, 0]
    acc_earth_y = a_earth[:, 1]

    # Angular velocity in device frame (from gyro or quat-deltas inside your impl)
    ang_velocity = _calculate_angular_velocity(rotation=rot)  # [T,3]

    # Spin/tilt w.r.t. gravity
    w_spin = np.einsum("td,td->t", ang_velocity, g_hat)
    w_tilt = np.linalg.norm(ang_velocity - w_spin[:, None] * g_hat, axis=1)

    # Tilt angle
    tilt = np.arccos(np.clip(g_hat[:, 2], -1.0, 1.0))

    # Quaternion geodesic rate
    delta = rot[:-1].inv() * rot[1:]                        # [T-1]
    rotvec = delta.as_rotvec()                              # [T-1,3]
    theta_step = np.linalg.norm(rotvec, axis=1)             # [T-1]
    theta_rate = np.pad(theta_step / dt, (1, 0))

    # Spin vs tilt rates from rotation axis vs mid-step gravity
    axis = rotvec / (theta_step[:, None] + eps)             # [T-1,3] unit
    g_mid = (g_hat[:-1] + g_hat[1:])
    g_mid /= (np.linalg.norm(g_mid, axis=1, keepdims=True) + eps)
    spin_frac_signed = np.sum(axis * g_mid, axis=1)         # [-1,1]
    theta_spin_rate = np.pad((theta_step * spin_frac_signed) / dt, (1, 0))
    theta_tilt_rate = np.pad(theta_step * np.sqrt(np.clip(1.0 - spin_frac_signed**2, 0.0, None)) / dt, (1, 0))

    # Jerk and component rates
    da = np.diff(a_dev, axis=0) / dt                        # [T-1,3]
    jerk_rate = np.pad(np.linalg.norm(da, axis=1), (1, 0))
    acc_vertical_rate = np.pad(np.diff(a_vert), (1, 0)) / dt
    acc_horizontal_rate = np.pad(np.diff(a_horz), (1, 0)) / dt

    # NEW: second finite-difference magnitude of accel (|Δ²a|/dt²)
    dda = np.diff(da, axis=0) / dt                          # [T-2,3]
    acc_dd_mag_rate = np.pad(np.linalg.norm(dda, axis=1), (2, 0))

    # NEW: short-window std of horizontal accel
    pad = horiz_std_window // 2
    xpad = np.pad(a_horz, (pad, pad), mode="edge")
    # Build rolling [T, win] with stride tricks
    shape = (T, horiz_std_window)
    stride = xpad.strides[0]
    view = np.lib.stride_tricks.as_strided(xpad, shape=shape, strides=(stride, stride))
    mu = view.mean(axis=1)
    acc_horizontal_std_w = np.sqrt(np.clip((view**2).mean(axis=1) - mu**2, 0.0, None))


    # length features
    forearm_length: float = 0.01 * float(forearm_cm)
    acc_horizontal_over_L = a_horz / forearm_length
    acc_vertical_over_L = a_vert / forearm_length
    jerk_rate_over_L = jerk_rate / forearm_length

    omega = np.linalg.norm(ang_velocity, axis=1)                         # [T]
    alpha = np.zeros_like(omega)                              # [T], rad/s^2
    dw = np.diff(ang_velocity, axis=0) / dt_sec
    alpha[1:] = np.linalg.norm(dw, axis=1)

    a_cen_over_g = (omega ** 2) * forearm_length / 9.8
    a_tan_over_g = alpha * forearm_length / 9.8
    v_nd = omega * np.sqrt(forearm_length / 9.8)
    
    names = [
        "x_earth_x",  # 3
        "x_earth_y",  # 3
        "x_earth_z",  # 3
        "y_earth_x",  # 3
        "y_earth_y",  # 3
        "y_earth_z",  # 3
        "ang_velocity_x",  # 3
        "ang_velocity_y",  # 3
        "ang_velocity_z",  # 3
        "a_vert",  # 1
        "a_horz",  # 1
        "w_spin",  # 1
        "w_tilt",  # 1
        "tilt",  # 1
        "theta_rate",  # 1
        "jerk_rate",  # 1
        "acc_earth_x",  # 1  *
        "acc_earth_y",  # 1  *
        "theta_spin_rate",  # 1  *
        "theta_tilt_rate",  # 1  *
        "acc_vertical_rate",  # 1  *
        "acc_horizontal_rate",  # 1  *
        "acc_dd_mag_rate",  # 1  *
        "acc_horizontal_std_w",  # 1  *
        "acc_horizontal_over_L",
        "acc_vertical_over_L",
        "jerk_rate_over_L",
        "a_cen_over_g",
        "a_tan_over_g",
        "v_nd"
    ]
    feat = np.concatenate(
        [
            x_earth,                              # 3
            y_earth,                              # 3
            ang_velocity,                         # 3
            a_vert[:, None],                      # 1
            a_horz[:, None],                      # 1
            w_spin[:, None],                      # 1
            w_tilt[:, None],                      # 1
            tilt[:, None],                        # 1
            theta_rate[:, None],                  # 1
            jerk_rate[:, None],                   # 1
            acc_earth_x[:, None],                 # 1  *
            acc_earth_y[:, None],                 # 1  *
            theta_spin_rate[:, None],             # 1  *
            theta_tilt_rate[:, None],             # 1  *
            acc_vertical_rate[:, None],           # 1  *
            acc_horizontal_rate[:, None],         # 1  *
            acc_dd_mag_rate[:, None],             # 1  *
            acc_horizontal_std_w[:, None],        # 1  *
            acc_horizontal_over_L[:, None],
            acc_vertical_over_L[:, None],
            jerk_rate_over_L[:, None],
            a_cen_over_g[:, None],
            a_tan_over_g[:, None],
            v_nd[:, None]
        ],
        axis=1,
    )

    return pd.DataFrame(feat, columns=names)

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
    orientation_features = [f'rot_{x}' for x in ('w', 'x', 'y', 'z')]

    additional_features = []
    for sequence_idx, sequence_id in tqdm(enumerate(sequence_ids), total=len(sequence_ids)):
        subject_id = data.loc[sequence_id]['subject'].iloc[0]

        features = extract_features(
            orientation=data.loc[sequence_id, orientation_features].values,
            linear_acceleration=data.loc[sequence_id, acc_features].values,
            forearm_cm=subject_meta.loc[subject_id]['elbow_to_wrist_cm']
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


    train = polars.scan_parquet(
        source=input_path,
    ).filter(polars.col("sequence_id").is_in(train_sequence_ids)).collect()

    val = polars.scan_parquet(
        source=input_path,
    ).filter(polars.col("sequence_id").is_in(val_sequence_ids)).collect()

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


    C = polars.read_parquet(
        source=input_path / 'train.parquet' / f'sequence_id={meta[0]["sequence_id"]}',
    ).shape[-1] - 1
    sum_ = np.zeros(C, dtype=np.float64)
    sumsq_ = np.zeros(C, dtype=np.float64)
    N = 0

    sequence_ids = [x['sequence_id'] for x in meta]
    for i, seq_id in tqdm(enumerate(sequence_ids), total=len(sequence_ids)):
        seq = polars.read_parquet(
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

    column_names = polars.read_parquet(
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
