import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None


def _window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, c)


def _window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class ConditionedMambaMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_scale=1.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        mod_scale_init=0.0,
    ):
        super().__init__()
        if selective_scan_fn is None:
            raise ImportError("mamba_ssm selective_scan_fn is required for MSSM-lite fusion.")

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            bias=conv_bias,
            padding=d_conv // 2,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            bias=conv_bias,
            padding=d_conv // 2,
        )

        self.depth_proj = nn.Linear(self.d_model, self.dt_rank + self.d_state * 2, bias=False)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner // 2) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        a = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner // 2)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2))
        self.D._no_weight_decay = True

        self.alpha_dt = nn.Parameter(torch.full((1,), mod_scale_init))
        self.alpha_B = nn.Parameter(torch.full((1,), mod_scale_init))
        self.alpha_C = nn.Parameter(torch.full((1,), mod_scale_init))

    def forward(self, x_rgb, x_dep):
        _, seqlen, _ = x_rgb.shape

        xz = self.in_proj(x_rgb)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        x_rgb_proj = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        x_dep_proj = self.depth_proj(x_dep.reshape(-1, x_dep.shape[-1]))

        dt_rgb, b_rgb, c_rgb = torch.split(x_rgb_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_dep, b_dep, c_dep = torch.split(x_dep_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = dt_rgb + self.alpha_dt * dt_dep
        b = b_rgb * (1.0 + self.alpha_B * torch.tanh(b_dep))
        c = c_rgb * (1.0 + self.alpha_C * torch.tanh(c_dep))

        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        b = rearrange(b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        c = rearrange(c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        a = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x,
            dt,
            a,
            b,
            c,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=None,
        )
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class ConditionedMambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        d_state=8,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        mod_scale_init=0.0,
    ):
        super().__init__()
        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_dep = nn.LayerNorm(dim)
        self.mixer = ConditionedMambaMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            mod_scale_init=mod_scale_init,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x_rgb, x_dep):
        x_rgb = x_rgb + self.drop_path(self.mixer(self.norm_rgb(x_rgb), self.norm_dep(x_dep)))
        x_rgb = x_rgb + self.drop_path(self.mlp(self.norm_mlp(x_rgb)))
        return x_rgb


class WindowConditionedMambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth=1,
        window_size=14,
        d_state=8,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        mod_scale_init=0.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList(
            [
                ConditionedMambaBlock(
                    dim=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    mod_scale_init=mod_scale_init,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, rgb_2d, dep_2d):
        b, h, w, c = rgb_2d.shape
        ws = min(self.window_size, h, w)

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            rgb_2d = F.pad(rgb_2d.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            dep_2d = F.pad(dep_2d.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        hp, wp = rgb_2d.shape[1], rgb_2d.shape[2]

        rgb_windows = _window_partition(rgb_2d, ws)
        dep_windows = _window_partition(dep_2d, ws)
        for block in self.blocks:
            rgb_windows = block(rgb_windows, dep_windows)

        out = _window_reverse(rgb_windows, ws, hp, wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :h, :w, :].contiguous()
        return out


class TemporalMambaBlock(nn.Module):
    def __init__(self, dim, d_state=4, expand=1):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm Mamba is required for temporal modeling.")

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            expand=expand,
        )

    def forward(self, x_seq):
        if x_seq.dim() != 4:
            raise ValueError(f"TemporalMambaBlock expects [B, T, L, C], got {tuple(x_seq.shape)}")

        b, t, l, c = x_seq.shape
        x = x_seq.transpose(1, 2).contiguous().view(b * l, t, c)
        x = x + self.mamba(self.norm(x))
        return x.view(b, l, t, c).transpose(1, 2).contiguous()
