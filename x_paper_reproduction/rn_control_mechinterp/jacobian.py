from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import copy

import numpy as np
import torch


@dataclass
class JacobianSpectrum:
    singular_values: np.ndarray
    right_vectors: np.ndarray
    rn_cosines: np.ndarray
    point: str

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            singular_values=self.singular_values,
            right_vectors=self.right_vectors,
            rn_cosines=self.rn_cosines,
            point=np.asarray([self.point]),
        )


def _orthogonalize(v: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    for q in basis:
        v = v - torch.dot(v, q) * q
    return v / v.norm().clamp_min(1e-12)


def top_input_singular_directions(
    *,
    runner,
    images: torch.Tensor,
    k: int = 4,
    iterations: int = 7,
    point: str = "trained",
    seed: int = 20260901,
) -> JacobianSpectrum:
    """Randomized power iteration on J^T J for RN -> surviving post-B13 state.

    The B13 block is deep-copied to FP32 so this diagnostic is numerically stable
    and does not mutate the loaded model.  The output contains all surviving
    tokens for the supplied image batch.
    """
    device = images.device
    x_pre = runner.pre_b13_state(images).float().detach()
    blk = copy.deepcopy(runner.visual.transformer.resblocks[runner.insert_block]).float().to(device).eval()
    trained = runner.visual.read_null_token.detach().float().to(device)
    if point == "trained":
        r0 = trained.clone()
    elif point == "zero":
        r0 = torch.zeros_like(trained)
    else:
        raise ValueError("point must be 'trained' or 'zero'")

    def f(r: torch.Tensor) -> torch.Tensor:
        tok = r.view(1, 1, -1).expand(1, x_pre.shape[1], -1)
        inp = torch.cat([x_pre, tok], dim=0)
        out = blk(inp)
        return out[:-1].permute(1, 0, 2).reshape(-1)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    basis: list[torch.Tensor] = []
    sigmas: list[float] = []

    for _comp in range(int(k)):
        v = torch.randn(r0.shape, generator=gen, device=device, dtype=torch.float32)
        v = _orthogonalize(v, basis)
        for _ in range(int(iterations)):
            # Jv
            _, jv = torch.autograd.functional.jvp(
                f, (r0.detach().requires_grad_(True),), (v,), create_graph=False, strict=True
            )
            # J^T(Jv)
            r = r0.detach().requires_grad_(True)
            y = f(r)
            (jtjv,) = torch.autograd.grad(y, r, grad_outputs=jv.detach(), retain_graph=False, create_graph=False)
            v = _orthogonalize(jtjv.detach(), basis)

        _, jv = torch.autograd.functional.jvp(
            f, (r0.detach().requires_grad_(True),), (v,), create_graph=False, strict=True
        )
        sigma = float(jv.norm().detach().cpu())
        basis.append(v.detach())
        sigmas.append(sigma)

    V = torch.stack(basis, dim=0)
    trained_unit = trained / trained.norm().clamp_min(1e-12)
    rn_cos = V @ trained_unit
    return JacobianSpectrum(
        singular_values=np.asarray(sigmas, dtype=np.float64),
        right_vectors=V.detach().cpu().numpy(),
        rn_cosines=rn_cos.detach().cpu().numpy(),
        point=point,
    )
