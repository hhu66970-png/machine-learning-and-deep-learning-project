"""PEFT methods on a timm ViT-B/16.

All methods share one recipe: freeze the pretrained backbone, then train only
a small set of (inserted or selected) parameters + the classifier head.

Supported methods:
    full_ft       - train everything (upper-bound baseline)
    linear_probe  - train only the classifier head (lower-bound baseline)
    bitfit        - train only bias terms + head
    lora          - low-rank adapters on attention q,v + head
    adaptformer   - parallel bottleneck adapter on the MLP branch + head
    ssf           - per-channel scale & shift modulation + head
    vpt           - learnable prompt tokens prepended to the sequence + head
    lora_ssf      - (our improvement) LoRA + SSF combined + head
"""
import math
import types
import torch
import torch.nn as nn
import timm

METHODS = [
    "full_ft", "linear_probe", "bitfit",
    "lora", "adaptformer", "ssf", "vpt", "lora_ssf",
]

DEFAULT_MODEL = "vit_base_patch16_224.augreg_in21k"


def build_model(num_classes, model_name=DEFAULT_MODEL, drop_path=0.0):
    return timm.create_model(
        model_name, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path
    )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class LoRAqkv(nn.Module):
    """Wrap a frozen fused qkv Linear, add low-rank deltas to q and v."""

    def __init__(self, qkv: nn.Linear, r=8, alpha=8, dropout=0.0):
        super().__init__()
        self.qkv = qkv
        dim = qkv.in_features
        self.dim = dim
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.lora_A_q = nn.Linear(dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv[..., :self.dim], qkv[..., self.dim:2 * self.dim], qkv[..., 2 * self.dim:]
        q = q + self.lora_B_q(self.lora_A_q(self.drop(x))) * self.scaling
        v = v + self.lora_B_v(self.lora_A_v(self.drop(x))) * self.scaling
        return torch.cat([q, k, v], dim=-1)


class Adapter(nn.Module):
    """AdaptFormer-style parallel bottleneck: up(GELU(down(x))) * s."""

    def __init__(self, dim, bottleneck=64, scale_init=0.1, dropout=0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * scale_init)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.drop(self.act(self.down(x)))) * self.scale


class SSF(nn.Module):
    """Per-channel scale & shift:  y = x * gamma + beta."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    def forward(self, x):
        return x * self.scale + self.shift


# --------------------------------------------------------------------------- #
# Patched block / model forwards
# --------------------------------------------------------------------------- #
def _adaptformer_block_forward(self, x):
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    h = self.norm2(x)
    x = x + self.drop_path2(self.ls2(self.mlp(h))) + self.adapter(h)
    return x


def _ssf_block_forward(self, x):
    a = self.ssf_attn(self.attn(self.ssf_n1(self.norm1(x))))
    x = x + self.drop_path1(self.ls1(a))
    m = self.ssf_mlp(self.mlp(self.ssf_n2(self.norm2(x))))
    x = x + self.drop_path2(self.ls2(m))
    return x


def _make_vpt_forward_features(num_prompts):
    def forward_features(self, x, *args, **kwargs):  # accept/ignore attn_mask etc.
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        B = x.shape[0]
        prompts = self.vpt_prompt.expand(B, -1, -1)
        x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)  # insert after cls token
        x = self.blocks(x)
        x = self.norm(x)
        return x
    return forward_features


# --------------------------------------------------------------------------- #
# Injectors
# --------------------------------------------------------------------------- #
def inject_lora(model, r=8, alpha=8, dropout=0.0):
    for blk in model.blocks:
        blk.attn.qkv = LoRAqkv(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)


def inject_adapter(model, bottleneck=64, scale_init=0.1, dropout=0.0):
    dim = model.embed_dim
    for blk in model.blocks:
        blk.adapter = Adapter(dim, bottleneck=bottleneck, scale_init=scale_init, dropout=dropout)
        blk.forward = types.MethodType(_adaptformer_block_forward, blk)


def inject_ssf(model, with_attn_mlp=True):
    dim = model.embed_dim
    for blk in model.blocks:
        blk.ssf_n1 = SSF(dim)
        blk.ssf_n2 = SSF(dim)
        blk.ssf_attn = SSF(dim) if with_attn_mlp else nn.Identity()
        blk.ssf_mlp = SSF(dim) if with_attn_mlp else nn.Identity()
        blk.forward = types.MethodType(_ssf_block_forward, blk)


def inject_vpt(model, num_prompts=20):
    dim = model.embed_dim
    prompt = nn.Parameter(torch.zeros(1, num_prompts, dim))
    nn.init.trunc_normal_(prompt, std=0.02)
    model.vpt_prompt = prompt
    model.forward_features = types.MethodType(_make_vpt_forward_features(num_prompts), model)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
_UNFREEZE_PATTERNS = {
    "linear_probe": [],
    "bitfit": [".bias"],
    "lora": ["lora_"],
    "adaptformer": ["adapter"],
    "ssf": ["ssf_"],
    "vpt": ["vpt_prompt"],
    "lora_ssf": ["lora_", "ssf_"],
}


def apply_peft(model, method, cfg=None):
    """Modify `model` in place for the given PEFT method; set requires_grad."""
    cfg = cfg or {}
    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    if method not in _UNFREEZE_PATTERNS:
        raise ValueError(f"unknown method: {method}")

    # 1) inject method-specific modules
    if method in ("lora", "lora_ssf"):
        inject_lora(model, r=cfg.get("lora_r", 8), alpha=cfg.get("lora_alpha", 8),
                    dropout=cfg.get("lora_dropout", 0.0))
    if method == "adaptformer":
        inject_adapter(model, bottleneck=cfg.get("adapter_dim", 64),
                       scale_init=cfg.get("adapter_scale", 0.1))
    if method in ("ssf", "lora_ssf"):
        inject_ssf(model)
    if method == "vpt":
        inject_vpt(model, num_prompts=cfg.get("vpt_prompts", 20))

    # 2) freeze everything, then unfreeze by name pattern + head
    for p in model.parameters():
        p.requires_grad_(False)
    patterns = _UNFREEZE_PATTERNS[method]
    for n, p in model.named_parameters():
        if any(pat in n for pat in patterns):
            p.requires_grad_(True)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    return model
