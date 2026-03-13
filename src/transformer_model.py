import torch.nn as nn
import torch
from transformer_blocks import *


class Encoder(nn.Module):
    """ Create an instance for encoder component.

    Create a encoder with multiple encoder block.

    Attributes:
        layers: A ModuleList of EncoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the encoder layer.

        Args:
            layers: A ModuleList of EncoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm  = LayerNormalization()

    def forward(self, x, mask):
        """ Forward function for encoder layer.

        This function will pass the input through multiple encoder block and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder layer.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """ Create an instance for decoder component.

    Create a decoder with multiple decoder block.

    Attributes:
        layers: A ModuleList of DecoderBlock layers.
        norm: A LayerNormalization layer.
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        """ Initialize the decoder layer.

        Args:
            layers: A ModuleList of DecoderBlock layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """ Forward function for decoder layer.

        This function will pass the input through multiple decoder block, and perform the layer normalization.

        Args:
            x: A tensor representing the input.
            encoder_output: A tensor representing the output of the encoder layer.
            src_mask: A tensor representing the source mask.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder layer.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """ Create an instance for transformer model.

    Create a transformer model with encoder, decoder, source embedding, target embedding, source positional encoding, target positional encoding, and projection layer.

    Attributes:
        encoder: A Encoder layer.
        decoder: A Decoder layer.
        src_embed: A InputEmbedding layer.
        tgt_embed: A InputEmbedding layer.
        src_pos: A PositionalEncoding layer.
        tgt_pos: A PositionalEncoding layer.
        projection_layer: A ProjectionLayer layer.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """ Initialize the transformer model.

        Args:
            encoder: A Encoder layer.
            decoder: A Decoder layer.
            src_embed: A InputEmbedding layer.
            tgt_embed: A InputEmbedding layer.
            src_pos: A PositionalEncoding layer.
            tgt_pos: A PositionalEncoding layer.
            projection_layer: A ProjectionLayer layer.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        self.relu = nn.ReLU()

    def encode(self, src, src_mask):
        """ Encode the source sequence.

        This function will embed the source sequence, add the positional encoding, and feed it into the encoder.

        Args:
            src: A tensor representing the source sequence.
            src_mask: A tensor representing the source mask.

        Returns:
            A tensor representing the output of the encoder.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """ Decode the target sequence.

        This function will embed the target sequence, add the positional encoding, and feed it into the decoder.

        Args:
            encoder_output: A tensor representing the output of the encoder.
            src_mask: A tensor representing the source mask.
            tgt: A tensor representing the target sequence.
            tgt_mask: A tensor representing the target mask.

        Returns:
            A tensor representing the output of the decoder.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """ Project the output of the decoder to the target vocabulary.

        This function will apply the projection layer to the output of the decoder, and return the result.

        Args:
            x: A tensor representing the output of the decoder.

        Returns:
            A tensor representing the output of the projection layer.
        """
        # return self.projection_layer(x)
        return torch.log_softmax(x @ self.tgt_embed.embedding.weight.T, dim=-1)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """Build a transformer model.

    Args:
        src_vocab_size (int): The number of unique words in the source language.
        tgt_vocab_size (int): The number of unique words in the target language.
        src_seq_len (int): The length of the sequence in the source language.
        tgt_seq_len (int): The length of the sequence in the target language.
        d_model (int, optional): The dimensionality of the model. Defaults to 512.
        N (int, optional): The number of encoder and decoder layers. Defaults to 6.
        h (int, optional): The number of heads in the multi-head attention. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimensionality of the feed forward layer. Defaults to 2048.

    Returns:
        Transformer: The transformer model.
    """
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer