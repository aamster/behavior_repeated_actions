import torch
import torch.nn.functional as F


class CollateFunction:
    def __init__(self, pad_token_id: int = 0, fixed_length: int = 64):
        self._pad_token_id = pad_token_id
        self._fixed_length = fixed_length

    def __call__(self, batch):
        x, y, sequence_y = zip(*batch)
        x_padded = torch.nn.utils.rnn.pad_sequence(
            x, batch_first=True, padding_value=self._pad_token_id
        )
        x_padded = F.pad(
            input=x_padded,
            pad=(0, 0, 0, self._fixed_length - x_padded.shape[1]),
            value=self._pad_token_id,
        )

        y_padded = torch.nn.utils.rnn.pad_sequence(
            y, batch_first=True, padding_value=self._pad_token_id
        )
        y_padded = F.pad(
            input=y_padded,
            pad=(0, self._fixed_length - y_padded.shape[1]),
            value=self._pad_token_id,
        )

        sequence_labels = torch.tensor(sequence_y, dtype=torch.long)
        return x_padded, y_padded, sequence_labels
