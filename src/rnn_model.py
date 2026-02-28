import torch
import torch.nn as nn


# inspired by https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BHW2RNNDecoder(nn.Module):
    def __init__(self, hidden_dim, dataset, rnn_type = nn.RNN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.embeddings = nn.Embedding(self.dataset.vocab_size, hidden_dim)
        self.rnn = rnn_type(hidden_dim, hidden_dim, batch_first=True)
        self.linear_head = nn.Linear(hidden_dim, self.dataset.vocab_size)


    def forward(self, encoder_hidden, target_idx):
        # print(encoder_hidden.shape, target_idx.shape)
        decoder_hidden = encoder_hidden
        targets = self.embeddings(target_idx)
        # print(targets.shape, decoder_hidden.shape)
        decoder_output, decoder_hidden = self.rnn(targets, decoder_hidden)
        # print(decoder_output.shape, decoder_hidden.shape)
        decoder_output = self.linear_head(decoder_output)
        # print(decoder_output.shape)

        return decoder_output, decoder_hidden


class BHW2RNNEncoder(nn.Module):
    def __init__(self, hidden_dim, dataset, rnn_type = nn.RNN):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.embeddings = nn.Embedding(self.dataset.vocab_size, embedding_dim=hidden_dim)
        self.rnn = rnn_type(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embeds = self.embeddings(x)
        output, hidden = self.rnn(embeds)
        return output, hidden



class BHW2RNNModel(nn.Module):
    def __init__(self, de_dataset, en_dataset, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.de_dataset = de_dataset
        self.en_dataset = en_dataset
        self.enc = BHW2RNNEncoder(hidden_dim, dataset=de_dataset)
        self.dec = BHW2RNNDecoder(hidden_dim, en_dataset)


    def forward(self, de_tokens, en_tokens):
        enc_output, enc_hidden = self.enc(de_tokens)
        # print(enc_hidden.shape)
        logits, dec_hidden = self.dec(enc_hidden, en_tokens)
        # print(logits.shape)
        return logits
        

    @torch.inference_mode()
    def inference(self, input):
        pass
