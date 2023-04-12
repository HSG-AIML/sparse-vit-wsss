import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import lightning.pytorch as pl


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ConcreteGate(pl.LightningModule):
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Usage example: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param l2_penalty: coefficient on the regularizer that minimizes l2 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    :param hard: if True, gates are binarized to {0, 1} but backprop is still performed as if they were concrete
    :param local_rep: if True, samples a different gumbel noise tensor for each sample in batch,
        by default, noise is sampled using shape param as size.
    """

    def __init__(
        self,
        shape,
        device,
        temperature=0.33,
        stretch_limits=(-0.1, 1.1),
        l0_penalty=0.0,
        eps=1e-6,
        hard=False,
        local_rep=False,
    ):
        super().__init__()

        self.temperature, self.stretch_limits, self.eps = (
            temperature,
            stretch_limits,
            eps,
        )
        self.l0_penalty = l0_penalty
        self.hard, self.local_rep = hard, local_rep
        self.log_a = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(shape))
        )

    def forward(self, x):
        """applies gate to values, if is_train, adds regularizer to reg_collection"""
        gates, _ = self.get_gates(shape=x.shape if self.local_rep else None)
        return x * gates

    def get_gates(self, shape=None):
        """samples gate activations in [0, 1] interval"""
        low, high = self.stretch_limits
        if self.training:
            shape = self.log_a.shape if shape is None else shape
            self.noise = torch.empty(shape).type_as(self.log_a)
            self.noise.uniform_(self.eps, 1.0 - self.eps)
            concrete = torch.sigmoid(
                (torch.log(self.noise) - torch.log(1 - self.noise) + self.log_a)
                / self.temperature
            )
        else:
            concrete = torch.sigmoid(self.log_a)
        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)

        pre_clipped_concrete = clipped_concrete
        if self.hard:
            hard_concrete = torch.gt(clipped_concrete, 0.5).to(torch.float)
            clipped_concrete = (
                clipped_concrete + (hard_concrete - clipped_concrete).detach()
            )
        return clipped_concrete, pre_clipped_concrete

    def get_penalty(self, values=None, axis=None):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        low, high = self.stretch_limits
        assert (
            low < 0.0
        ), "p_gate_closed can be computed only if lower stretch limit is negative"

        p_open = torch.sigmoid(
            self.log_a - self.temperature * torch.log(torch.tensor(-low / high))
        )
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        if values is not None:
            p_open += torch.zeros_like(values)  # broadcast shape to account for values

        l0_reg = self.l0_penalty * torch.sum(p_open)
        return torch.mean(l0_reg)

    def get_sparsity_rate(self):
        """Computes the fraction of gates which are now active (non-zero)"""
        is_nonzero = torch.ne(self.get_gates()[0], 0.0)
        return torch.mean(is_nonzero.to(torch.float))


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, dim, device, prune, heads=8, dim_head=64, dropout=0.0, l0_penalty=0.2
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.prune = prune
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.prune:
            self.gate = ConcreteGate(
                (1, heads, 1, 1),
                device=device,
                l0_penalty=l0_penalty,
                hard=True,
                local_rep=False,
            )

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        if self.prune:
            attn = self.gate(attn)

        out_pre_fusion = torch.matmul(attn, v)
        out = rearrange(out_pre_fusion, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        device,
        prune,
        dropout=0.0,
        l0_penalty=0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                device,
                                prune,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                l0_penalty=l0_penalty,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x_att = attn(x)
            x = x_att + x
            x = ff(x) + x
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class HeadSeg(nn.Module):
    def __init__(self, features, nclasses=2):
        super(HeadSeg, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=7, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, nclasses, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        device,
        prune=False,
        seg=False,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        l0_penalty=0.2,
        multimodal=False
    ):
        super().__init__()

        self.depth = depth
        self.multimodal = multimodal

        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        self.seg = seg
        assert (
            self.image_height % self.patch_height == 0
            and self.image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )
        patch_dim = channels * self.patch_height * self.patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_height,
                p2=self.patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.sigmoid = nn.Sigmoid()

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, device, prune, dropout, l0_penalty
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        if self.seg:
            self.seg_head = HeadSeg(257, nclasses=num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        if self.seg:
            b, n, _ = x.shape
            x = x.reshape((b, n, 32, 32))
            x = self.seg_head(x)
        else:
            x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
            x = self.to_latent(x)
            x = self.mlp_head(x)

        return self.sigmoid(x)


class MViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        device,
        seg=False,
        prune=False,
        pool="cls",
        m1_channels=2,
        m2_channels=13,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        l0_penalty=0.2,
        multimodal=True
    ):
        super().__init__()

        self.depth = depth
        self.multimodal = multimodal

        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)

        self.seg = seg
        assert (
            self.image_height % self.patch_height == 0
            and self.image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (self.image_height // self.patch_height) * (
            self.image_width // self.patch_width
        )
        m1_patch_dim = m1_channels * self.patch_height * self.patch_width
        m2_patch_dim = m2_channels * self.patch_height * self.patch_width

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.m1_to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_height,
                p2=self.patch_width,
            ),
            nn.Linear(m1_patch_dim, dim),
        )

        self.m2_to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_height,
                p2=self.patch_width,
            ),
            nn.Linear(m2_patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.sigmoid = nn.Sigmoid()

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, device, prune, dropout, l0_penalty
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        if self.seg:
            self.seg_head = HeadSeg(257, nclasses=num_classes)

    def forward(self, m1, m2):
        x1 = self.m1_to_patch_embedding(m1)
        x2 = self.m2_to_patch_embedding(m2)
        x = x1 + x2

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 n d -> b n d", b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        if self.seg:
            b, n, _ = x.shape
            x = x.reshape((b, n, 32, 32))
            x = self.seg_head(x)
        else:
            x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
            x = self.to_latent(x)
            x = self.mlp_head(x)

        return self.sigmoid(x)
