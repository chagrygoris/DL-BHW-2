from dataset import BHW2Dataset, BHW2Allin1Dataset
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import warnings
warnings.filterwarnings("ignore")


def create_dataset(split : str, path_to_data="../data"):
    de = "{}/{}.de-en.de".format(path_to_data, split)
    de_dataset = BHW2Dataset(de)
    if split == "test1":
        return de_dataset

    en = "{}/{}.de-en.en".format(path_to_data, split)
    en_dataset = BHW2Dataset(en)
    return BHW2Allin1Dataset(de_dataset, en_dataset)


def create_dataloaders(path_to_data="../data", batch_size=32):
    train_set = create_dataset("train", path_to_data=path_to_data)
    val_set = create_dataset("val", path_to_data=path_to_data)
    test_set = create_dataset("test1", path_to_data=path_to_data)
    val_set.en.vocab, val_set.en.vocab_size = train_set.en.vocab, train_set.en.vocab_size
    val_set.de.vocab, val_set.de.vocab_size = train_set.de.vocab, train_set.de.vocab_size
    test_set.vocab = train_set.de.vocab
    test_set.vocab_size = train_set.de.vocab_size
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


from typing import Union
import torch
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device : Union[torch.device, str] ="cpu"):
    model.train()
    model.to(device)
    total_loss = 0
    num_tokens = 0

    for de, de_lengths, en, en_lenghts in tqdm(loader):
        de_tokens = de[:, :de_lengths.max()].to(device)
        en_tokens = en[:, :en_lenghts.max()].to(device)
        # print(de_tokens.shape, en_tokens.shape)
        optimizer.zero_grad()
        logits = model(de_tokens, en_tokens[:, :-1])
        loss = criterion(logits.permute(0, 2, 1), en_tokens[:, 1:])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(en_tokens)
        num_tokens += len(en_tokens)

    return total_loss / num_tokens


@torch.no_grad()
def validate_epoch(model, loader, criterion, device : Union[torch.device, str] ="cpu"):
    model.eval()
    model.to(device)
    total_loss = 0
    num_tokens = 0

    for de, de_lengths, en, en_lenghts in tqdm(loader):
        de_tokens = de[:, :de_lengths.max()].to(device)
        en_tokens = en[:, :en_lenghts.max()].to(device)
        logits = model(de_tokens, en_tokens[:, :-1])
        loss = criterion(logits.permute(0, 2, 1), en_tokens[:, 1:])

        total_loss += loss.item() * len(en_tokens)
        num_tokens += len(en_tokens)

    return total_loss / num_tokens


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, n_epochs : int = 1, device : Union[torch.device, str] = "cpu"):
    for i in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss = validate_epoch(model, val_loader, criterion, device=device)
        scheduler.step()
        print("Training epoch {} / {} : train_loss {}, val_loss {}".format(i, n_epochs, train_loss, val_loss))
