import numpy as np
import torch
import re


def get_p2i(data):
    """
    Get the patient to index mapping.
    """

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def log_value_stats(data, values, min_count=100):
    """
    Per-token (mean, std) of log(value) over rows that carry a value (value > 0), for get_batch's
    z-scoring. Token ids are raw (before get_batch's +1 shift). Tokens with fewer than min_count
    values are left out and get z = 0.
    """
    has_value = values > 0
    tokens = data[has_value, 2]
    logs = np.log(values[has_value].astype(np.float64))
    stats = {}
    for tok_id in np.unique(tokens):
        vals = logs[tokens == tok_id]
        if len(vals) > min_count:
            stats[int(tok_id)] = (float(vals.mean()), float(vals.std()))
    return stats


def get_batch(ix, data, p2i, select='random', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False, 
              no_event_token_rate=5, cut_batch=False,
              values_data=None, values_stats=None):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'left', 'right', or 'random' (window taken from the start, end, or a random offset of the trajectory)
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible
        values_data: numpy array of float32 numeric values parallel to data (one value per row), or None
        values_stats: dict of {token_id: (mean, std)} of log values (see log_value_stats), or None

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages
        v: numeric values (B, T, 2) with [z_score, has_value] — only returned if values_data is not None
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    # Extract numeric values if available
    has_values = values_data is not None
    if has_values:
        raw_vals = torch.from_numpy(values_data[batch_idx].astype(np.float32))
        raw_vals = raw_vals.masked_fill(~mask, 0.0)

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    # insert a "no event" token every 5 years on average
    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values

    # augment lifestyle tokens to avoid immortality bias
    # NOTE: must happen AFTER computing m, otherwise augmented ages inflate m
    # and let too many no-event padding tokens survive, hurting calibration
    if lifestyle_augmentations:
        lifestyle_idx = (tokens >= 3) * (tokens <= 11)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    # Also extend values with zeros for "no event" padding tokens
    if has_values:
        raw_vals = torch.hstack([raw_vals, torch.zeros(len(ix), pad.shape[1])])

    # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)
    if has_values:
        raw_vals = raw_vals.masked_fill(ages > m, 0.0)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)
    if has_values:
        raw_vals = torch.gather(raw_vals, 1, s)

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if has_values:
            raw_vals = raw_vals[:, cut_margin:]

    # cut to maintain the block size
    #TODO it would be better to use the strategy defined by the "select" parameter
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if has_values:
            raw_vals = raw_vals[:, cut_margin:]

    # shift by one to generate targets
    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]
    if has_values:
        v = raw_vals[:, :-1]  # values align with input tokens, not targets

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)
    if has_values:
        v = v.masked_fill(x == 0, 0.0)

    # normalize values as z-scores of log(value) per token type, since lab values are positive and
    # often skewed (CRP, ALT, creatinine); the second channel marks which tokens carry a value,
    # since a lab result exactly at its mean has z = 0
    if has_values:
        z_scores = torch.zeros_like(v)
        has_value = torch.zeros_like(v)
        if values_stats is not None:
            # x contains token IDs (already shifted by +1), so original token IDs are x-1
            for tok_id, (mean, std) in values_stats.items():
                tok_mask = (x == tok_id + 1) & (v != 0)  # +1 because tokens are shifted
                if tok_mask.any() and std > 0:
                    z_scores[tok_mask] = (torch.log(v[tok_mask]) - mean) / std
                    has_value[tok_mask] = 1.0
        z_scores = torch.clamp(z_scores, -5.0, 5.0)
        v = torch.stack([z_scores, has_value], dim=-1)  # (B, T, 2)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
        if has_values:
            v = v.pin_memory().to(device, non_blocking=True)
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
        if has_values:
            v = v.to(device)

    if has_values:
        return x, a, y, b, v
    return x, a, y, b


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def shap_model_creator(model, disease_ids, person_tokens_ids, person_ages, device):
    """
    Creates a pseudo model that returns only logits for specified tokens.
    Needed for SHAP values, otherwise the SHAP visualisation is too huge.
    """
    def f(ps):
        xs = []
        as_ = []

        for p in ps:
            if len(p) == 0:
                print('No tokens found??')
                raise
            p = list(map(int, p))
            new_tokens = []
            new_ages = []
            for num, (masked, value, age) in enumerate(zip(p, person_tokens_ids, person_ages)):
                if num == 0:
                    new_ages.append(age)
                    if masked == 10000:
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_tokens.append(value)

            x = (torch.tensor(new_tokens, device=device)[None, ...])
            a = (torch.tensor(new_ages, device=device)[None, ...])

            xs.append(x)
            as_.append(a)

        max_length = max([x.shape[-1] for x in xs])

        xs = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs]
        as_ = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in as_]

        x = torch.cat(xs)
        a = torch.cat(as_)

        with torch.no_grad():
            probs = model(x, a)[0][:, -1, disease_ids].detach().cpu().numpy()
        return probs

    return f
