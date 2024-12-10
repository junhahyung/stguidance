import importlib.metadata
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "naive": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
    do_stg=False,
    txt_len=-1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        
        if do_stg:
            batch_size = q.shape[0]
            q, q_perturb = q[:batch_size-1], q[batch_size-1:]
            k, k_perturb = k[:batch_size-1], k[batch_size-1:]
            v, v_perturb = v[:batch_size-1], v[batch_size-1:]
            if attn_mask:
                attn_mask = attn_mask[:batch_size-1]
            seq_len = q.shape[2]
            attn_mask = torch.zeros((seq_len, seq_len), dtype=q_perturb.dtype, device="cuda")
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
            )
            batch_size = q_perturb.shape[0]
            seq_len = q_perturb.shape[2]
            num_heads = q_perturb.shape[1]
            identity_block_size = txt_len
            
            mask_start = seq_len - txt_len
            mask_indices = torch.arange(mask_start, seq_len, device="cuda")
            full_mask = torch.zeros((seq_len, seq_len), dtype=q_perturb.dtype, device="cuda")

            # txt_len 부분만 마스킹 적용
            full_mask[mask_indices[:, None], mask_indices] = float("-inf")
            full_mask[mask_indices, mask_indices] = 0  # 대각선 값 초기화

            # 필요한 부분만 확장
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            full_mask = full_mask.expand(batch_size, num_heads, seq_len, seq_len)

            # 최적화된 마스크를 사용하여 `x_perturb` 계산
            x_perturb = F.scaled_dot_product_attention(
                q_perturb, k_perturb, v_perturb, attn_mask=full_mask, dropout_p=drop_rate, is_causal=causal,
            )

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )

    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    elif mode == "naive":
        assert do_stg == True
        batch_size = q.shape[0]
        q, q_perturb = q[:batch_size-1], q[batch_size-1:]
        k, k_perturb = k[:batch_size-1], k[batch_size-1:]
        v, v_perturb = v[:batch_size-1], v[batch_size-1:]
        if attn_mask:
            attn_mask = attn_mask[:batch_size-1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
        batch_size = q_perturb.shape[0]
        seq_len = q_perturb.shape[2]
        num_heads = q_perturb.shape[1]
        identity_block_size = seq_len - txt_seq_len
        attn_scores_perturb = torch.matmul(q_perturb, k_perturb.transpose(-2, -1))
        d_k = q.size(-1)
        attn_scores_perturb = attn_scores_perturb / (d_k ** 0.5)
        attn_scores_perturb = attn_scores_perturb / torch.sqrt(torch.tensor(q.shape[-1], dtype=attn_scores_perturb.dtype, device=attn_scores_perturb.device))
        attn_weights = F.softmax(attn_scores_perturb, dim=-1)
        print(f"attn_weights shape: {attn_weights.shape}")
        print(f"v_perturb shape: {v_perturb.shape}")
        x_perturb = torch.matmul(attn_weights, v_perturb)
        x = torch.cat([x, x_perturb], dim=0)
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out
