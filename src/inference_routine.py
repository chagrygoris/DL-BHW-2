import torch
import torch.nn as nn


def causal_mask(seq_len, batch_size=1):
    mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0


@torch.no_grad()
def greedy_decode_b(model, source, source_mask, max_len=80, device="cuda"):
    batch_size = source.size(0)
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(batch_size, 1).fill_(2).type_as(source).to(device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.masked_fill(finished, 3)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
        finished |= next_word == 3
        if finished.all():
            break

    return decoder_input


@torch.no_grad()
def beam_decode_b(model, source, source_mask, beam_size=5, max_len=80, device="cuda", length_penalty_alpha=0.6):
    batch_size = source.size(0)
    k = beam_size
    encoder_output = model.encode(source, source_mask)
    lp_base = (5.0 + 1.0) ** length_penalty_alpha
    src_mask_rep = source_mask.repeat_interleave(k, dim=0)
    enc_rep = encoder_output.repeat_interleave(k, dim=0)

    seqs = torch.full((batch_size, k, 1), 2, dtype=source.dtype, device=device)
    log_probs = torch.full((batch_size, k), -1e9, device=device)
    log_probs[:, 0] = 0.0
    finished = torch.zeros(batch_size, k, dtype=torch.bool, device=device)

    for _ in range(1, max_len):
        t_len = seqs.size(2)
        decoder_input = seqs.view(batch_size * k, t_len)
        decoder_mask = causal_mask(t_len).type_as(source_mask).to(device)

        out = model.decode(enc_rep, src_mask_rep, decoder_input, decoder_mask)
        logp = model.project(out[:, -1]).view(batch_size, k, -1)

        logp = logp.masked_fill(finished.unsqueeze(-1), -1e9)
        eos_scores = logp[:, :, 3]
        eos_scores = torch.where(finished, torch.zeros_like(eos_scores), eos_scores)
        logp[:, :, 3] = eos_scores

        topk_logp, topk_idx = logp.topk(k, dim=-1)
        candidate_raw = (log_probs.unsqueeze(-1) + topk_logp).view(batch_size, -1)
        candidate_len = t_len + 1
        denom = ((5.0 + candidate_len) ** length_penalty_alpha) / lp_base
        candidate_norm = candidate_raw / denom

        topk_norm, topk_pos = candidate_norm.topk(k, dim=-1)

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, k)
        beam_from = topk_pos // k
        token_choice = topk_pos % k

        next_tokens = topk_idx[batch_idx, beam_from, token_choice]
        prev = seqs[batch_idx, beam_from]
        seqs = torch.cat([prev, next_tokens.unsqueeze(-1)], dim=2)
        log_probs = candidate_raw[batch_idx, topk_pos]
        finished = finished[batch_idx, beam_from] | (next_tokens == 3)

        if finished.all():
            break

    seq_len = seqs.size(2)
    eq_eos = seqs == 3
    has_eos = eq_eos.any(dim=2)
    first_pos = torch.argmax(eq_eos.to(torch.int64), dim=2)
    lengths = torch.where(has_eos, first_pos + 1, torch.full_like(first_pos, seq_len))

    denom_final = ((5.0 + lengths.float()) ** length_penalty_alpha) / lp_base
    final_scores = log_probs / denom_final

    best_beam = final_scores.argmax(dim=1)
    best_seq = seqs[torch.arange(batch_size, device=device), best_beam]
    return best_seq
