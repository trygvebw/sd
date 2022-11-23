from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


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
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
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
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
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
        b,c,h,w = q.shape
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


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., struct_attn=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.use_cross_attention_control = False
        self.last_attn_slice = None
        
        self.use_saved_s2 = False
        self.saved_s2 = None
        
        self.use_saved_r1 = False
        self.saved_r1 = None
        
        self.struct_attn = struct_attn
        
        self.callback1 = lambda *args: None
        self.callback2 = lambda *args: None
        
        self.default_mask = None
    
    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        
        if context is None:
            # Do not use mask for self-attention
            mask = None
        elif mask is None:
            mask = self.default_mask
        
        if self.struct_attn and isinstance(context, list):
            return self._struct_attn_forward(q, context, mask)
        elif isinstance(context, list):
            context = torch.cat([context[0], context[1]['k'][0]], dim=0) # use key tensor for context
        else:
            context = default(context, x)
        return self._forward(q, context, mask)
    
    def _struct_attn_forward(self, q, context, mask):
        """
        context: list of [uc, list of conditional context]
        """
        uc_context = context[0]
        context_k, context_v = context[1]['k'], context[1]['v']

        if isinstance(context_k, list) and isinstance(context_v, list):
            r2 = self._multi_qkv(q, uc_context, context_k, context_v, mask)
        elif isinstance(context_k, torch.Tensor) and isinstance(context_v, torch.Tensor):
            r2 = self._heterogenous_qkv(q, uc_context, context_k, context_v, mask)
        else:
            raise NotImplementedError

        return self.to_out(r2)

    def _get_kv(self, context):
        return self.to_k(context), self.to_v(context)
    
    def _multi_qkv(q, uc_context, context_k, context_v, mask):
        h = self.heads

        assert uc_context.size(0) == context_k[0].size(0) == context_v[0].size(0)
        true_bs = uc_context.size(0) * h

        k_uc, v_uc = self._get_kv(uc_context)
        k_c = [self.to_k(c_k) for c_k in context_k]
        v_c = [self.to_v(c_v) for c_v in context_v]
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        k_uc = rearrange(k_uc, 'b n (h d) -> (b h) n d', h=h)            
        v_uc = rearrange(v_uc, 'b n (h d) -> (b h) n d', h=h)

        k_c  = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c] # NOTE: modification point
        v_c  = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        # get composition
        sim_uc = einsum('b i d, b j d -> b i j', q[:true_bs], k_uc) * self.scale
        sim_c  = [einsum('b i d, b j d -> b i j', q[true_bs:], k) * self.scale for k in k_c]

        attn_uc = sim_uc.softmax(dim=-1)
        attn_c  = [sim.softmax(dim=-1) for sim in sim_c]

        # get uc output
        out_uc = einsum('b i j, b j d -> b i d', attn_uc, v_uc)

        # get c output
        if len(v_c) == 1:
            out_c_collect = []
            for attn in attn_c:
                for v in v_c:
                    out_c_collect.append(einsum('b i j, b j d -> b i d', attn, v))
            out_c = sum(out_c_collect) / len(out_c_collect)
        else:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)

        out = torch.cat([out_uc, out_c], dim=0)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  

        return out
    
    def _heterogenous_qkv(q, uc_context, context_k, context_v, mask):
        h = self.heads
        k = self.to_k(torch.cat([uc_context, context_k], dim=0))
        v = self.to_v(torch.cat([uc_context, context_v], dim=0))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return out

    def _forward(self, q_in, context, mask):
        h = self.heads
        
        if self.last_attn_slice is None:
            self.last_attn_slice = dict()
        
        if self.use_saved_r1:
            context = self.saved_r1
        else:
            self.saved_r1 = context

        k_in = self.to_k(context)
        v_in = self.to_v(context)
        del context

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

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
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        #print(q.shape, k.shape, v.shape)
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale
            
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(s1.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                s1.masked_fill_(~mask, max_neg_value)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            self.callback2(s1, s2)
            del s1
            
            if self.use_cross_attention_control:
                # From https://github.com/bloc97/CrossAttentionControl/blob/main/CrossAttention_Release.ipynb
                if self.use_last_attn_slice:
                    if self.last_attn_slice_mask is not None:
                        new_attn_slice = torch.index_select(self.last_attn_slice[i], -1, self.last_attn_slice_indices)
                        s2 = s2 * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                    else:
                        s2 = self.last_attn_slice[i]

                if self.save_last_attn_slice:
                    self.last_attn_slice[i] = s2

                if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                    s2 = s2 * self.last_attn_slice_weights
            
            if self.use_saved_s2:
                s2 = self.saved_s2
            else:
                self.saved_s2 = s2

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            
            del s2
        
        self.callback1(q, k, v)
        
        # Reset attention saving
        self.use_last_attn_slice = False
        self.save_last_attn_slice = False
        self.use_last_attn_weights = False

        del q, k, v

        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        return self.to_out(r2)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        return checkpoint(self._forward, (x, context, mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in