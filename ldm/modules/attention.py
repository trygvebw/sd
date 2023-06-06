import math

from inspect import isfunction
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from ldm.modules.tome import compute_merge
#from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.state_manager import DiffusionStateManager

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


from torch._dynamo import allow_in_graph

allow_in_graph(rearrange)
allow_in_graph(repeat)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., state_mgr=None):
        super().__init__()
        self.state_mgr = DiffusionStateManager.use_or_create(state_mgr)

        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = w_.softmax(dim=2, dtype=w_.dtype)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


@dataclass
class CrossAttentionControlState:
    last_attn_slice: dict = None
    use_last_attn_slice: bool = False
    save_last_attn_slice: bool = False
    use_last_attn_weights: bool = False

    use_saved_s2: bool = False
    saved_s2: torch.Tensor = None
    use_saved_r1: bool = False
    saved_r1: torch.Tensor = None

    def get_cac_s2(self, s2, slice_idx):
        # From https://github.com/bloc97/CrossAttentionControl/blob/main/CrossAttention_Release.ipynb
        if self.use_last_attn_slice:
            if self.last_attn_slice_mask is not None:
                new_attn_slice = torch.index_select(self.last_attn_slice[slice_idx], -1, self.last_attn_slice_indices)
                s2 = s2 * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
            else:
                s2 = self.last_attn_slice[slice_idx]

        if self.save_last_attn_slice:
            self.last_attn_slice[slice_idx] = s2

        if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
            s2 = s2 * self.last_attn_slice_weights

        return s2


class CrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, st_id=None,
                 attn_type='torch_flash', state_mgr=None):
        """
            attn_type: one of 'torch_flash' (default), 'xformers', 'vanilla'
        """
        super().__init__()
        self.state_mgr = DiffusionStateManager.use_or_create(state_mgr)

        self.query_dim = query_dim
        self.inner_dim = dim_head * heads
        self.context_dim = default(context_dim, self.query_dim)

        self.is_self_attn = self.query_dim == self.context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.context_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.query_dim),
            nn.Dropout(dropout))

        self.gamma_q = nn.Parameter(torch.ones(self.inner_dim))
        self.gamma_k = nn.Parameter(torch.ones(self.inner_dim))
        self.gamma_v = nn.Parameter(torch.ones(self.inner_dim))
        self.gamma_out = nn.Parameter(torch.ones(self.inner_dim))

        self.st_id = st_id
        self.attn_type = attn_type
        self.xformers_attention_op = None

        # Cross-attention control
        self.use_cross_attention_control = False
        self.cac_state = CrossAttentionControlState()

        self.disable = False

    def forward(self, x, context=None, mask=None, attn_type_override=None, **kwargs):
        if self.disable:
            return x

        attn_type = self.attn_type if attn_type_override is None else attn_type_override

        if context is None:
            # Do not use mask for self-attention
            mask = None

        if isinstance(context, list):
            context = torch.cat([context[0], context[1]['k'][0]], dim=0)  # use key tensor for context
        else:
            context = default(context, x)

        if self.use_cross_attention_control:
            if self.cac_state.last_attn_slice is None:
                self.cac_state.last_attn_slice = dict()

            if self.cac_state.use_saved_r1:
                context = self.cac_state.saved_r1
            else:
                self.cac_state.saved_r1 = context

        if self.use_cross_attention_control and attn_type != 'vanilla':
            raise Exception('Cross-attention control can only be used with attention type "vanilla"')

        q_in = self.gamma_q * self.to_q(x)
        k_in = self.gamma_k * self.to_k(context)
        v_in = self.gamma_v * self.to_v(context)
        del context

        rearrange_qkv = lambda y: rearrange(y, 'b n (h d) -> (b h) n d', h=self.heads)
        q, k, v = rearrange_qkv(q_in), rearrange_qkv(k_in), rearrange_qkv(v_in)
        del q_in, k_in, v_in

        match attn_type:
            case 'torch_flash':
                return self.torch_flash_forward(q, k, v, mask=mask, **kwargs)
            case 'xformers':
                return self.xformers_forward(q, k, v, mask=mask, **kwargs)
            case 'vanilla':
                return self.vanilla_forward(q, k, v, mask=mask, **kwargs)
            case _:
                raise Exception(f'{attn_type} is not a valid attention type')

    def vanilla_forward(self, q, k, v, mask=None, extra_conds=None, **kwargs):
        self.state_mgr.send('attention_forward_start')

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

        attn_scale = self.dim_head ** -0.5
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            #s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * attn_scale
            s1 = torch.baddbmm(
                torch.empty(q.shape[0], slice_size, k.shape[1], dtype=q.dtype, device=q.device),
                q[:, i:end],
                k.transpose(-1, -2),
                beta=0,
                alpha=attn_scale,
            )

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(s1.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
                s1.masked_fill_(~mask, max_neg_value)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            #self.state_mgr.send('attention_forward_softmax_piece', s2.detach().clone().cpu())

            if self.use_cross_attention_control:
                # Cross-attention control
                s2 = self.cac_state.get_cac_s2(s2, i)

                if self.cac_state.use_saved_s2:
                    s2 = self.cac_state.saved_s2
                else:
                    self.cac_state.saved_s2 = s2

            #r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            r1[:, i:end] = torch.bmm(s2, v)

            del s2

        # Reset attention saving
        self.cac_state.use_last_attn_slice = False
        self.cac_state.save_last_attn_slice = False
        self.cac_state.use_last_attn_weights = False

        del q, k, v

        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=self.heads)
        del r1

        to_out, out_dropout = self.to_out[0], self.to_out[1]
        r2 = self.gamma_out * to_out(r2)
        return out_dropout(r2)

    def torch_flash_forward(self, q, k, v, mask=None, upcast_attn=True, mem_efficient=False, callback=None, **kwargs):
        q_seq_length, inner_dim = q.shape[1:]
        kv_seq_length = k.shape[1]

        # scaled_dot_product_attention expects q, k, v on the format (batch_size, heads, seq_len, inner_dim)
        q = q.view(-1, self.heads, q_seq_length, inner_dim)
        k = k.view(-1, self.heads, kv_seq_length, inner_dim)
        v = v.view(-1, self.heads, kv_seq_length, inner_dim)

        batch_size = q.shape[0]

        if mask is not None:
            mask = self.prepare_attention_mask(mask, q_seq_length, batch_size)
            mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

        dtype = q.dtype
        if upcast_attn:
            q, k = q.float(), k.float()

        # the output of sdp = (batch, num_heads, seq_len, dim_head)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=mem_efficient):
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )

        if callback is not None:
            callback(self.__class__.__name__, {
                'is_self_attn': self.is_self_attn,
                'st_id': self.st_id,
                'q': q,
                'k': k,
                'v': v,
                'hidden_states': hidden_states
            })

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = self.gamma_out * self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

    def xformers_forward(self, q, k, v, mask=None, callback=None, **kwargs):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        batch_size = q.shape[0] // self.heads

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v,
                                                      attn_bias=mask,
                                                      op=self.xformers_attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(batch_size, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, out.shape[1], self.heads * self.dim_head)
        )

        to_out, out_dropout = self.to_out[0], self.to_out[1]
        out = self.gamma_out * to_out(out)
        out = out_dropout(out)

        if callback is not None:
            callback(self.__class__.__name__, {
                'is_self_attn': self.is_self_attn,
                'st_id': self.st_id,
                'q': q,
                'k': k,
                'v': v,
                'hidden_states': out
            })

        return out


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, st_id=None, attn_opts=None, state_mgr=None):
        super().__init__()
        self.state_mgr = DiffusionStateManager.use_or_create(state_mgr)
        self.disable_self_attn = disable_self_attn
        if attn_opts is None:
            attn_opts = {}

        # Is a self-attention if not self.disable_self_attn:
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None, st_id=st_id,
            state_mgr=self.state_mgr, **attn_opts)

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, state_mgr=self.state_mgr)

        # Is self-attn if context is none
        self.attn2 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim, st_id=st_id,
            state_mgr=self.state_mgr, **attn_opts)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, tome_options=None, callback=None, **kwargs):
        # No need for checkpointing in this inference-only ldm codebase:
        #return checkpoint(self._forward, (x, context, mask, tome_options, callback), self.parameters(), self.checkpoint)
        return self._forward(x, context=context, mask=mask, tome_options=tome_options, callback=callback, **kwargs)

    def _forward(self, x, context=None, mask=None, tome_options=None, callback=None, extra_conds=None, **kwargs):
        if tome_options is not None and tome_options.get('enabled', False):
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, tome_options)

            x = x + u_a(self.attn1(
                m_a(self.norm1(x)),
                context=context if self.disable_self_attn else None,
                mask=mask if self.disable_self_attn else None,
                callback=callback,
                **kwargs))
            x = x + u_c(self.attn2(
                m_c(self.norm2(x)),
                context=context,
                mask=mask,
                callback=callback,
                extra_conds=extra_conds,
                **kwargs))
            x = x + u_m(self.ff(m_m(self.norm3(x))))
            return x
        else:
            x = x + self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                mask=mask if self.disable_self_attn else None,
                callback=callback,
                **kwargs)
            x = x + self.attn2(
                self.norm2(x),
                context=context,
                mask=mask,
                callback=callback,
                extra_conds=extra_conds,
                **kwargs)
            x = x + self.ff(self.norm3(x))
            return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, st_id='unnamed',
                 state_mgr=None, disable=False):
        super().__init__()
        self.state_mgr = DiffusionStateManager.use_or_create(state_mgr)
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, st_id=st_id,
                                   state_mgr=self.state_mgr)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

        self.st_id = st_id
        self.disable = disable

    def forward(self, x, context=None, mask=None, callback=None, **kwargs):
        if self.disable:
            return x

        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            #context_i = torch.randint(high=len(context), size=(1,)).item()
            x = block(x, context=context[i], mask=mask, callback=callback, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
