import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class Attention(nn.Module):
    """Bahdanau attention mechanism"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_dim) - current decoder hidden state
        # encoder_outputs: (batch, src_len, hidden_dim) - all encoder outputs
        
        batch_size, src_len, hidden_dim = encoder_outputs.shape
        
        # Repeat decoder hidden state src_len times
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attention(
            torch.cat((decoder_hidden, encoder_outputs), dim=2)
        ))  # (batch, src_len, hidden_dim)
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        attention_weights = F.softmax(attention, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class BHW2AttnRNNDecoder(nn.Module):
    def __init__(self, hidden_dim, dataset, rnn_type=nn.GRU, max_seq_len: int = 128, 
                 device: Union[torch.device, str] = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embeddings = nn.Embedding(self.dataset.vocab_size, hidden_dim)
        
        # Attention mechanism
        self.attention = Attention(hidden_dim)
        
        # RNN takes concatenated input: [embedding, context]
        self.rnn = rnn_type(hidden_dim * 2, hidden_dim, batch_first=True)
        
        # Output layer
        self.linear_head = nn.Linear(hidden_dim * 2, self.dataset.vocab_size)
        
    def forward(self, encoder_outputs, encoder_hidden, target_idx):
        # encoder_outputs: (batch, src_len, hidden_dim)
        # encoder_hidden: (1, batch, hidden_dim) for GRU
        # target_idx: (batch, tgt_len) - includes BOS
        
        batch_size = target_idx.size(0)
        tgt_len = target_idx.size(1)
        
        # Get embeddings
        targets = self.embeddings(target_idx)  # (batch, tgt_len, hidden_dim)
        
        # Initialize decoder hidden state with encoder's final hidden
        decoder_hidden = encoder_hidden  # (1, batch, hidden_dim)
        
        # Store outputs
        outputs = []
        attentions = []
        
        # Get the first decoder hidden state for initial context
        decoder_hidden_squeezed = decoder_hidden.squeeze(0)  # (batch, hidden_dim)
        context, _ = self.attention(decoder_hidden_squeezed, encoder_outputs)
        
        # Loop through target sequence
        for t in range(tgt_len):
            # Get current target embedding
            target_t = targets[:, t:t+1, :]  # (batch, 1, hidden_dim)
            
            # Concatenate with context
            rnn_input = torch.cat([target_t, context.unsqueeze(1)], dim=-1)  # (batch, 1, hidden_dim*2)
            
            # RNN forward
            output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)  # output: (batch, 1, hidden_dim)
            
            # Update decoder hidden for next iteration
            decoder_hidden_squeezed = decoder_hidden.squeeze(0)
            
            # Compute new context with updated hidden state
            context, attention_weights = self.attention(decoder_hidden_squeezed, encoder_outputs)
            
            # Concatenate output with context for final prediction
            output_with_context = torch.cat([output.squeeze(1), context], dim=-1)  # (batch, hidden_dim*2)
            
            # Generate logits
            logits = self.linear_head(output_with_context)  # (batch, vocab_size)
            outputs.append(logits.unsqueeze(1))
            attentions.append(attention_weights.unsqueeze(1))
        
        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # (batch, tgt_len, vocab_size)
        attentions = torch.cat(attentions, dim=1)  # (batch, tgt_len, src_len)
        
        return outputs, decoder_hidden, attentions
    
    def inference(self, encoder_outputs, encoder_hidden):
        # encoder_outputs: (1, src_len, hidden_dim) for single batch
        # encoder_hidden: (1, 1, hidden_dim) for GRU with batch=1
        
        generated_seq = [self.dataset.bos_token]
        
        # Initial hidden state
        decoder_hidden = encoder_hidden
        
        # Get initial context
        decoder_hidden_squeezed = decoder_hidden.squeeze(0)  # (1, hidden_dim)
        context, _ = self.attention(decoder_hidden_squeezed, encoder_outputs)
        
        while len(generated_seq) < self.max_seq_len and generated_seq[-1] != self.dataset.eos_token:
            # Prepare input token
            current_token = torch.LongTensor([generated_seq[-1]]).to(self.device)
            
            # Embed
            embed = self.embeddings(current_token).unsqueeze(1)  # (1, 1, hidden_dim)
            
            # Concatenate with context
            rnn_input = torch.cat([embed, context.unsqueeze(1)], dim=-1)  # (1, 1, hidden_dim*2)
            
            # RNN forward
            output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
            
            # Update decoder hidden for next iteration
            decoder_hidden_squeezed = decoder_hidden.squeeze(0)
            
            # Compute new context
            context, attention_weights = self.attention(decoder_hidden_squeezed, encoder_outputs)
            
            # Concatenate output with context
            output_with_context = torch.cat([output.squeeze(1), context], dim=-1)
            
            # Generate logits
            logits = self.linear_head(output_with_context)
            
            # Get next token
            next_token = logits.argmax(dim=-1).item()
            generated_seq.append(next_token)
        
        return generated_seq


class BHW2AttnRNNEncoder(nn.Module):
    def __init__(self, hidden_dim, dataset, rnn_type=nn.GRU, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.bidirectional = bidirectional
        
        # Adjust hidden size for bidirectional
        self.encoder_hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        
        self.embeddings = nn.Embedding(self.dataset.vocab_size, embedding_dim=hidden_dim)
        self.rnn = rnn_type(
            hidden_dim, 
            self.encoder_hidden_dim, 
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Linear layer to project encoder hidden to decoder hidden size
        if bidirectional:
            self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.hidden_projection = nn.Identity()

    def forward(self, x):
        # x: (batch, src_len)
        embeds = self.embeddings(x)  # (batch, src_len, hidden_dim)
        output, hidden = self.rnn(embeds)
        
        # Handle bidirectional GRU hidden state
        if self.bidirectional:
            # hidden is (num_layers * 2, batch, hidden_dim)
            # Combine forward and backward last states
            hidden = hidden.view(-1, 2, hidden.size(1), self.encoder_hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1)
            # Project to decoder hidden size
            hidden = self.hidden_projection(hidden)
        
        # output: (batch, src_len, hidden_dim * num_directions)
        # For decoder, we need to project output if bidirectional
        if self.bidirectional:
            # Project output to decoder hidden size
            output = self.hidden_projection(output)
        
        return output, hidden


class BHW2AttnRNNModel(nn.Module):
    def __init__(self, de_dataset, en_dataset, hidden_dim=128, 
                 device: Union[torch.device, str] = "cpu",
                 rnn_type=nn.GRU,
                 bidirectional_encoder=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.de_dataset = de_dataset
        self.en_dataset = en_dataset
        self.device = device
        
        # Encoder with optional bidirectionality
        self.enc = BHW2AttnRNNEncoder(
            hidden_dim, 
            dataset=de_dataset, 
            rnn_type=rnn_type,
            bidirectional=bidirectional_encoder
        )
        
        # Decoder with attention
        self.dec = BHW2AttnRNNDecoder(
            hidden_dim, 
            en_dataset, 
            rnn_type=rnn_type,
            device=device
        )

    def forward(self, de_tokens, en_tokens):
        # de_tokens: (batch, src_len)
        # en_tokens: (batch, tgt_len) - includes BOS
        
        # Encoder forward
        enc_output, enc_hidden = self.enc(de_tokens)
        # enc_output: (batch, src_len, hidden_dim)
        # enc_hidden: (1, batch, hidden_dim) for unidirectional GRU
        
        # Decoder forward with attention
        logits, dec_hidden, attentions = self.dec(enc_output, enc_hidden, en_tokens)
        # logits: (batch, tgt_len, vocab_size)
        
        return logits  # Maintain same interface - return only logits

    @torch.inference_mode()
    def inference(self, de_tokens):
        # de_tokens: (1, src_len) for single sentence
        
        # Encoder forward
        enc_output, enc_hidden = self.enc(de_tokens)
        
        # Decoder inference with attention
        return self.dec.inference(enc_output, enc_hidden)