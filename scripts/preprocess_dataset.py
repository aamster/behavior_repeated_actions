import json
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import tensorstore
from tqdm import tqdm

from bfrb.dataset import ACTION_ID_MAP

CHANNEL_IDX_MAP = {
    **dict(zip([f'acc_{x}' for x in ('x', 'y', 'z')], range(0, 3))),
    **dict(zip([f'rot_{x}' for x in ('w', 'x', 'y', 'z')], range(3, 7))),

}

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

def write_data(input_data_path: Path, out_path: Path):
    data = pd.read_csv(input_data_path)
    data = data.set_index('sequence_id')
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

        for col_name, channel_idx in CHANNEL_IDX_MAP.items():
            values = data.loc[sequence_id][col_name].values
            preprocessed[sequence_idx, :len(values), channel_idx] = values

        if np.isnan(preprocessed[sequence_idx][:]).any():
            continue

        actions = []
        gesture = data.loc[sequence_id]['gesture'].iloc[0]
        for i, behavior in enumerate(data.loc[sequence_id]['behavior']):
            if behavior == 'Performs gesture':
                actions.append(ACTION_ID_MAP[gesture.lower()])
            else:
                actions.append(ACTION_ID_MAP[behavior.lower()])

        if not 'Performs gesture' in data.loc[sequence_id]['behavior'].unique():
            continue

        gesture_start = np.where((data.loc[sequence_id]['behavior'] == 'Performs gesture').values)[0][0]

        meta = {
            'sequence_id': sequence_id,
            'arr_idx': sequence_idx,
            'actions': actions,
            'gesture_start': int(gesture_start),
            'sequence_length': len(data.loc[sequence_id])
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
    rng = np.random.default_rng(seed=1234)
    splitter = DataSplitter(n_examples=data.shape[0], train_frac=train_frac, rng=rng)
    train_idxs, val_idxs = splitter.split()

    with open(input_path / 'meta.json') as f:
        meta = json.load(f)

    arr_idxs = set([x['arr_idx'] for x in meta])

    train_idxs = [x for x in train_idxs if x in arr_idxs]
    val_idxs = [x for x in val_idxs if x in arr_idxs]

    train_meta = [x for x in meta if x['arr_idx'] in train_idxs]
    val_meta = [x for x in meta if x['arr_idx'] in val_idxs]

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

@click.command()
@click.option(
    '--input-data-path',
    type=click.Path(path_type=Path, readable=True)
)
@click.option(
    '--out-path',
    type=click.Path(path_type=Path, writable=True)
)
def main(input_data_path: Path, out_path: Path):
    #write_data(input_data_path=input_data_path, out_path=out_path)
    write_train_test_split(
        input_path=out_path,
        out_path=out_path.parent
    )

if __name__ == '__main__':
    main()
