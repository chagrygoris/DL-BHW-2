from dataset import BHW2Dataset, BHW2Allin1Dataset
from inference_routine import greedy_decode_b, beam_decode_b
import sacrebleu
from torch.utils.data import DataLoader
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import warnings
warnings.filterwarnings("ignore")
import torch
from typing import Union
import torch
from tqdm import tqdm


def create_decoder_causal_mask(batch_size, seq_len):
    mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0

def collate_fn(batch, device = torch.device("cuda")):
    src = torch.cat(list(map(lambda x : x[0].unsqueeze(0), batch)))
    src_lens = torch.cat(list(map(lambda x : x[1].unsqueeze(0), batch)))
    src = src[:, :src_lens.max()]
    src_mask = (src != 1).unsqueeze(1).unsqueeze(1)
    if len(batch[0]) == 2:
        return {
            "src" : src,
            "src_mask" : src_mask
        }

    tgt = torch.cat(list(map(lambda x : x[2].unsqueeze(0), batch)))
    tgt_lens = torch.cat(list(map(lambda x : x[3].unsqueeze(0), batch)))
    tgt = tgt[:, :tgt_lens.max() - 1]
    labels = torch.cat(list(map(lambda x : x[2].unsqueeze(0), batch)))
    labels = labels[:, 1:tgt_lens.max()]
    tgt_mask = ((tgt != 1) & (tgt != 3)).unsqueeze(1) & create_decoder_causal_mask(*tgt.shape).to(device)
    tgt_mask = tgt_mask.unsqueeze(1)
    return {
        "src" : src,
        "src_mask" : src_mask,
        "tgt" : tgt,
        "tgt_mask" : tgt_mask,
        "label" : labels
    }


def create_dataset(split : str, path_to_data="../data", device=torch.device("cpu"), en_vocab=None, de_vocab=None):
    de = "{}/{}.de-en.de".format(path_to_data, split)
    de_dataset = BHW2Dataset(de, device=device, vocab=de_vocab)
    if split == "test1":
        return de_dataset

    en = "{}/{}.de-en.en".format(path_to_data, split)
    en_dataset = BHW2Dataset(en, device=device, vocab=en_vocab)
    return BHW2Allin1Dataset(de_dataset, en_dataset)


def create_dataloaders_tf(path_to_data="../data", batch_size=32, device=torch.device("cpu")):
    train_set = create_dataset("train", path_to_data=path_to_data, device=device)
    val_set = create_dataset("val", path_to_data=path_to_data, device=device, en_vocab=train_set.en.vocab, de_vocab=train_set.de.vocab)
    test_set = create_dataset("test1", path_to_data=path_to_data, device=device, en_vocab=train_set.en.vocab, de_vocab=train_set.de.vocab)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=lambda batch : collate_fn(batch, device=device))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda batch : collate_fn(batch, device=device))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda batch : collate_fn(batch, device=device))
    return train_loader, val_loader, test_loader


def create_dataloaders(path_to_data="../data", batch_size=32, device=torch.device("cpu")):
    train_set = create_dataset("train", path_to_data=path_to_data, device=device)
    val_set = create_dataset("val", path_to_data=path_to_data, device=device, en_vocab=train_set.en.vocab, de_vocab=train_set.de.vocab)
    test_set = create_dataset("test1", path_to_data=path_to_data, device=device, en_vocab=train_set.en.vocab, de_vocab=train_set.de.vocab)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, scheduler, device = torch.device("cuda")):
    model.train()
    model.to(device)
    total_loss = 0
    numel = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model.encode(batch["src"], batch["src_mask"])
        out = model.decode(out, batch["src_mask"], batch["tgt"], batch["tgt_mask"])
        out = model.project(out)
        loss = criterion(out.transpose(1, 2), batch["label"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * batch["tgt"].shape[0]
        numel += batch["tgt"].shape[0]

    return total_loss / numel


@torch.no_grad()
def evaluate_b(model, val_loader, dataset, test=False, use_beam_search=False):
    hyps = []
    refs = []
    model.eval()
    for batch in val_loader:
        if not use_beam_search:
            hyp = greedy_decode_b(model, batch["src"], batch["src_mask"])
        else:
            hyp = beam_decode_b(model, batch["src"], batch["src_mask"])
        for sequence in hyp:
            hyps.append(
                list(filter(lambda x : x > 3, sequence.tolist()))
            )
        if test:
            continue
        ref = batch["tgt"]
        for sequence in ref:
            refs.append(
                list(filter(lambda x : x > 3, sequence.tolist()))
            )
    hyps = list(map(lambda x : dataset.idx2token(x), hyps))
    if test:
        return hyps
    refs = list(map(lambda x : dataset.idx2token(x), refs))
    return hyps, refs

@torch.no_grad()
def validation_epoch(model, val_loader, dataset, use_beam_search=False):
    hyps, refs = evaluate_b(model, val_loader, dataset, use_beam_search=use_beam_search)
    hyps_for_bleu = list(map(lambda x : ' '.join(x), hyps))
    refs_for_bleu = [list(map(lambda x : ' '.join(x), refs))]
    bleu = sacrebleu.corpus_bleu(hyps_for_bleu, refs_for_bleu, tokenize="none", force=True)
    return bleu.score



def train_tf(model, train_loader, val_loader, optimizer, criterion, scheduler, device, n_epochs=10):
    for i in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        val_bleu = validation_epoch(model, val_loader, train_loader.dataset.en)
        print(f"Epoch {i} / {n_epochs} : train x-entropy : {train_loss}, validation BLEU : {val_bleu}")
