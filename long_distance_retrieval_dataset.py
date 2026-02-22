import torch
from torch.utils.data import Dataset, DataLoader


class LongDistanceRetrievalDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int = 512, vocab_size: int = 128):
        if seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        if vocab_size != 128:
            raise ValueError("vocab_size must be 128 for this task")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.NEEDLE_ID = 126
        self.QUERY_ID = 127
        self.random_low = 1
        self.random_high_exclusive = 121

    def _sample_tokens_excluding_v(self, v: int, size: tuple[int, ...]) -> torch.Tensor:
        base = torch.randint(1, 120, size=size, dtype=torch.long)
        return base + (base >= v).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        value_token = torch.randint(
            low=self.random_low,
            high=self.random_high_exclusive,
            size=(1,),
            dtype=torch.long,
        ).item()

        input_ids = self._sample_tokens_excluding_v(value_token, (self.seq_len,))

        first_half_end = max(1, self.seq_len // 2)
        needle_pos = torch.randint(0, first_half_end, (1,), dtype=torch.long).item()

        input_ids[needle_pos] = self.NEEDLE_ID
        input_ids[needle_pos + 1] = value_token
        input_ids[-1] = self.QUERY_ID

        target_ids = torch.full((self.seq_len,), -100, dtype=torch.long)
        target_ids[-1] = value_token

        return input_ids, target_ids


if __name__ == "__main__":
    dataset = LongDistanceRetrievalDataset(num_samples=8, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for input_ids, target_ids in dataloader:
        print(input_ids.shape, target_ids.shape)
        print(input_ids[0, -5:])
        print(target_ids[0, -5:])
        break