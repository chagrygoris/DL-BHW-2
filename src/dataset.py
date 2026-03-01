import io
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from typing import List
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import warnings
warnings.filterwarnings("ignore")

# https://docs.pytorch.org/text/stable/vocab.html#build-vocab-from-iterator
def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield line.strip().split()


class BHW2Dataset(Dataset):
    def __init__(self, file_path, max_seq_len=128, sanity_checker=False):
        super().__init__()
        self.file_path = file_path
        self.sanity_checker = sanity_checker
        self.max_seq_len = max_seq_len
        self.texts = list(yield_tokens(file_path))
        self.vocab = build_vocab_from_iterator(yield_tokens(file_path), specials=["<unk>", "<pad>", "<bos>", "<eos>"], max_tokens=50000)
        self.vocab_size = len(self.vocab)
        self.pad_token = self.vocab["<pad>"]
        self.unk_token = self.vocab["<unk>"]
        self.bos_token = self.vocab["<bos>"]
        self.eos_token = self.vocab["<eos>"]
        self.vocab.set_default_index(self.unk_token)

    def token2idx(self, tokens : List[str]):
        return self.vocab.lookup_indices(tokens)

    def idx2token(self, indices : List[int]):
        return self.vocab.lookup_tokens(indices)

    def __len__(self):
        if self.sanity_checker:
            return 128
        return 128
        return len(self.texts)

    def __getitem__(self, index):
        out = torch.full((self.max_seq_len, ), fill_value=self.pad_token)
        tokens = ["<bos>"] + self.texts[index] + ["<eos>"]
        seq_len = len(tokens)
        tokens = self.token2idx(tokens)
        tokens = torch.tensor(tokens)
        out[:len(tokens)] = tokens
        return out, seq_len


class BHW2Allin1Dataset(Dataset):
    def __init__(self, de_dataset, en_dataset):
        self.de = de_dataset
        self.en = en_dataset

    def __len__(self):
        return self.de.__len__()

    def __getitem__(self, i):
        de_i = self.de[i]
        en_i = self.en[i]
        return de_i[0], de_i[1], en_i[0], en_i[1]
