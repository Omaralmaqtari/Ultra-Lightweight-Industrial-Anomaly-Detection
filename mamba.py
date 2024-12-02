"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum


class Mamba(nn.Module):
    def __init__(self, ch_in, mid_ch, patchsize, d_state):
        super(Mamba, self).__init__()
        n_layer = 2
        d_inner = patchsize
        dt_rank = math.ceil(d_inner / 16)
        
        self.input = nn.Conv2d(ch_in, mid_ch, kernel_size=8, stride=8, groups=ch_in//2)
        
        self.layers = nn.ModuleList([])
        for _ in range(n_layer):
            self.layers.append(nn.ModuleList([
                nn.GroupNorm(2,mid_ch),
                MambaBlock(mid_ch, d_inner, d_state, dt_rank),
                ]))
        
        self.output = nn.Conv2d(mid_ch, mid_ch, kernel_size=1, padding=0, groups=ch_in//2)
        
    def forward(self, x):
        x = self.input(x)
        for norm, mambablock in self.layers:
            x = mambablock(norm(x)) + x
            
        out = self.output(x)
        out = F.interpolate(out, scale_factor=(8), mode='bilinear')
        return out
    
    
class MambaBlock(nn.Module):
    def __init__(self, mid_ch, d_inner, d_state, dt_rank):
        super(MambaBlock, self).__init__()
        self.mid_ch = mid_ch
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        
        self.in_proj = nn.Conv2d(mid_ch, mid_ch * 2, kernel_size=3, padding=1, groups=mid_ch//2, bias=False)
        self.conv2d = nn.Conv2d(mid_ch, mid_ch, kernel_size=1, padding=0, groups=mid_ch//2, bias=True)
        
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
    def forward(self, x):
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.mid_ch, self.mid_ch], dim=1)
        x = self.conv2d(x)
        x = F.silu(x)
        
        _, _, h, _ = x.size()
        x = rearrange(x, 'b c h w -> b c (h w)')
        y = rearrange(self.ssm(x), 'b c (h w) -> b c h w', h=h)
        
        y = y * F.silu(res)

        return y
    
    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        y = y + u * D
        
        return y
        