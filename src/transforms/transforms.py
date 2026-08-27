import math
from abc import abstractmethod
from typing import Optional, List

import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

from .matrix import (
    GeneralMatrix,
    OrthogonalMatrix,
    SVDMatrix,
    l2norm_along_axis1, 
    sample_chi
)
from ..helpers import decompose_dim, split_dim
from ..utils.common_utils import filter_kwarg_dict


MATRIX_PARAMETRIZATIONS = {
    "general": GeneralMatrix,
    "orthogonal": OrthogonalMatrix,
    "svd": SVDMatrix, # TODO also general, but in different format - rename?
}


class BaseTransform(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        pass

    @abstractmethod
    def remove_parametrizations(self) -> None:
        pass


class IdentityTransform(BaseTransform):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        return x
    
    def remove_parametrizations(self) -> None:
        pass


class FullTransform(BaseTransform):

    def __init__(
        self, 
        size: int, 
        init: str = "orthogonal",
        parametrization: str = "general",
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.matrix = MATRIX_PARAMETRIZATIONS[parametrization](size, init, device, dtype)

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        t = self.matrix() if not inv_t else self.matrix.inv_t()
        return torch.tensordot(x, t, dims=((dim,), (0,)))
    
    def remove_parametrizations(self) -> None:
        self.matrix.remove_parametrizations()


class HadamardTransform(BaseTransform):

    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size
        self.scale = 1 / math.sqrt(self.group_size)

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # Hadamard transform is it own inverse
        x_shape = x.shape
        return hadamard_transform(x.view(-1, self.group_size), scale=self.scale).view(x_shape)
    
    def remove_parametrizations(self) -> None:
        pass


class GivensTransform(BaseTransform):
    """Data-aware block-diagonal Givens transform.

    The matrix is calibrated on the first activation tensor passed to ``forward``.
    Calls with ``inv_t=True`` do not trigger calibration, since those calls normally
    operate on weights.  Blocks without a massive outlier use a normalized Hadamard
    matrix, matching the baseline transform used by FP-Quant.
    """

    def __init__(
        self,
        size: Optional[int] = None,
        group_size: int = 32,
        n_iter: Optional[int] = None,
        outlier_threshold: float = 50.0,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        if group_size <= 1 or group_size & (group_size - 1):
            raise ValueError("Givens group_size must be a power of two greater than one")
        if size is not None and size % group_size:
            raise ValueError(f"Transform size {size} must be divisible by group_size {group_size}")
        if n_iter is not None and n_iter < 0:
            raise ValueError("n_iter must be non-negative")

        self.size = size
        self.group_size = group_size
        self.n_iter = group_size - 1 if n_iter is None else n_iter
        self.outlier_threshold = outlier_threshold
        self.matrix_device = device
        self.matrix_dtype = dtype
        self.register_buffer("mat", None)
        self._observed_vectors = None
        self._observed_maxima = None
        self.givens_blocks = 0
        self.hadamard_blocks = 0
        self.observed_abs_max = 0.0

    @staticmethod
    def _givens_rotation(
        n: int,
        i: int,
        j: int,
        theta: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        matrix = torch.eye(n, device=device, dtype=torch.float32)
        c, s = torch.cos(theta), torch.sin(theta)
        matrix[i, i] = c
        matrix[j, j] = c
        matrix[i, j] = -s
        matrix[j, i] = s
        return matrix

    @staticmethod
    def _closest_lower_level(standard: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        levels = standard * standard.new_tensor([2 / 3, 1 / 2, 1 / 6])
        valid = levels <= value.abs()
        return levels[valid].max() if valid.any() else value.abs()

    @staticmethod
    def _hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
        if torch.device(device).type == "cuda":
            return hadamard_transform(
                torch.eye(size, device=device, dtype=torch.float32),
                scale=1 / math.sqrt(size),
            )

        # fast-hadamard-transform is CUDA-only; keep calibration and tests usable
        # on CPU without changing the matrix convention used by the extension.
        matrix = torch.ones(1, 1, device=device, dtype=torch.float32)
        while matrix.shape[0] < size:
            matrix = torch.cat(
                (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
                dim=0,
            )
        return matrix / math.sqrt(size)

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        """Accumulate one representative high-magnitude row per channel block."""
        if x.ndim == 0:
            raise ValueError("Givens calibration requires at least one channel dimension")
        x_flat = x.reshape(-1, x.shape[-1]).float()
        full_size = x_flat.shape[-1]
        if self.size is not None and full_size != self.size:
            raise ValueError(f"Expected {self.size} channels, got {full_size}")
        if full_size % self.group_size:
            raise ValueError(
                f"Input size {full_size} must be divisible by group_size {self.group_size}"
            )

        num_groups = full_size // self.group_size
        if self._observed_vectors is None:
            self._observed_vectors = torch.zeros(
                num_groups, self.group_size, device="cpu", dtype=torch.float32
            )
            self._observed_maxima = torch.full((num_groups,), -torch.inf)

        for group_idx, start in enumerate(range(0, full_size, self.group_size)):
            group = x_flat[:, start:start + self.group_size]
            flat_index = group.abs().argmax()
            maximum = group.abs().flatten()[flat_index].cpu()
            if maximum > self._observed_maxima[group_idx]:
                row = torch.div(flat_index, self.group_size, rounding_mode="floor").item()
                self._observed_vectors[group_idx].copy_(group[row].cpu())
                self._observed_maxima[group_idx] = maximum

    @torch.no_grad()
    def finalize_calibration(self) -> None:
        """Build the transform after one or more calls to :meth:`observe`."""
        if self._observed_vectors is None:
            raise RuntimeError("No activation samples were observed for Givens calibration")
        num_groups = self._observed_vectors.shape[0]
        full_size = num_groups * self.group_size
        representatives = torch.zeros(num_groups, full_size, dtype=torch.float32)
        for group_idx in range(num_groups):
            start = group_idx * self.group_size
            representatives[group_idx, start:start + self.group_size] = self._observed_vectors[group_idx]
        self._build_matrix(representatives)
        self._observed_vectors = None
        self._observed_maxima = None

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        """Immediately calibrate from a single activation tensor."""
        x_flat = x.reshape(-1, x.shape[-1]).float()
        self._validate_size(x_flat.shape[-1])
        self._build_matrix(x_flat)

    def _validate_size(self, full_size: int) -> None:
        if self.size is not None and full_size != self.size:
            raise ValueError(f"Expected {self.size} channels, got {full_size}")
        if full_size % self.group_size:
            raise ValueError(
                f"Input size {full_size} must be divisible by group_size {self.group_size}"
            )

    def _build_matrix(self, x_flat: torch.Tensor) -> None:
        full_size = x_flat.shape[-1]
        self._validate_size(full_size)

        matrix_device = self.matrix_device or x_flat.device
        block_mats = []
        self.givens_blocks = 0
        self.hadamard_blocks = 0
        self.observed_abs_max = x_flat.abs().max().item()
        for start in range(0, full_size, self.group_size):
            group = x_flat[:, start:start + self.group_size]
            block = self._hadamard_matrix(self.group_size, matrix_device)
            flat_index = group.abs().argmax()
            row = torch.div(flat_index, self.group_size, rounding_mode="floor").item()
            current_col = (flat_index % self.group_size).item()
            vector = group[row].to(matrix_device).clone()

            if vector[current_col].abs() > self.outlier_threshold:
                self.givens_blocks += 1
                block = torch.eye(self.group_size, device=matrix_device, dtype=torch.float32)
                group_max = torch.exp2(torch.floor(torch.log2(vector[current_col].abs())) - 1)

                for _ in range(self.n_iter):
                    candidates = vector.abs().clone()
                    candidates[current_col] = torch.inf
                    other_col = candidates.argmin().item()
                    a, b = vector[current_col], vector[other_col]
                    if a.abs() < torch.finfo(torch.float32).eps:
                        break

                    target_magnitude = (
                        group_max if a.abs() > group_max
                        else self._closest_lower_level(group_max, a)
                    )
                    a_target = a.sign() * target_magnitude
                    energy = a.square() + b.square()
                    if a_target.square() > energy:
                        break
                    b_magnitude = torch.sqrt(torch.clamp_min(energy - a_target.square(), 0))
                    b_target = b.sign() * b_magnitude if b != 0 else b_magnitude

                    # For row vectors and the matrix convention below, atan2 retains
                    # the rotation direction that acos would lose.
                    dot = a * a_target + b * b_target
                    cross = b * a_target - a * b_target
                    theta = torch.atan2(cross, dot)
                    rotation = self._givens_rotation(
                        self.group_size, current_col, other_col, theta, matrix_device
                    )
                    vector = vector @ rotation
                    block = block @ rotation
                    current_col = other_col
            else:
                self.hadamard_blocks += 1

            block_mats.append(block)

        dtype = self.matrix_dtype or x_flat.dtype
        self.mat = torch.stack(block_mats).to(device=matrix_device, dtype=dtype)

    def to_matrix(self) -> torch.Tensor:
        """Materialize the full block-diagonal matrix (intended for tests/export)."""
        if self.mat is None:
            raise RuntimeError("Givens transform has not been calibrated")
        return torch.block_diag(*self.mat.unbind(0))

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if self.mat is None:
            if inv_t:
                return x
            self.calibrate(x.movedim(dim, -1))

        matrices = self.mat.to(device=x.device, dtype=x.dtype)
        dim = dim if dim >= 0 else x.ndim + dim
        moved = x.movedim(dim, -1)
        if moved.shape[-1] != matrices.shape[0] * self.group_size:
            raise ValueError(
                f"Expected {matrices.shape[0] * self.group_size} channels, got {moved.shape[-1]}"
            )
        grouped = moved.unflatten(-1, (matrices.shape[0], self.group_size))
        transformed = torch.einsum("...gi,gij->...gj", grouped, matrices).flatten(-2)
        return transformed.movedim(-1, dim)

    def remove_parametrizations(self) -> None:
        pass


class KroneckerFactorizedTransform(BaseTransform):

    def __init__(
        self, 
        size: int, 
        init: str = "orthogonal",
        parametrization: str = "general",
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.lsize, self.rsize = decompose_dim(size)
        self.lmatrix = MATRIX_PARAMETRIZATIONS[parametrization](self.lsize, init, device, dtype)
        self.rmatrix = MATRIX_PARAMETRIZATIONS[parametrization](self.rsize, init, device, dtype)
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if dim == -1:
            dim = x.ndim - 1
            
        l = self.lmatrix() if not inv_t else self.lmatrix.inv_t()
        r = self.rmatrix() if not inv_t else self.rmatrix.inv_t()
        x = split_dim(x, self.lsize, dim)
        x = torch.matmul(x.movedim(dim, -1), l).movedim(-1, dim)
        x = torch.matmul(x.movedim(dim + 1, -1), r).movedim(-1, dim + 1)
        return x.flatten(dim, dim + 1)
    
    def remove_parametrizations(self) -> None:
        self.lmatrix.remove_parametrizations()
        self.rmatrix.remove_parametrizations()


class BlockDiagonalTransform(BaseTransform):
    pass

class IdentityLowRankTransform(BaseTransform):

    def __init__(
        self, 
        size: int, 
        rank: int,
        alpha: Optional[float] = None,
        device: torch.device = None, 
        dtype: torch.dtype = None
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha or rank
        self.lora_A = nn.Parameter(torch.empty(rank, size, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(size, rank, device=device, dtype=dtype)) 
        # Following LoRA paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if dim == -1:
            dim = x.ndim - 1

        if inv_t:
            # Woodbury Matrix identity
            inv = torch.eye(self.rank, device=x.device, dtype=x.dtype) + self.lora_A.mm(self.lora_B)
            res = torch.tensordot(x, self.lora_A.T, dims=((dim,), (0,)))
            res = torch.tensordot(res, inv, dims=((dim,), (0,)))
            res = -torch.tensordot(res, self.lora_B.T, dims=((dim,), (0,)))
        else:
            res = torch.tensordot(x, self.lora_A.T, dims=((dim,), (0,)))
            res = torch.tensordot(res, self.lora_B.T, dims=((dim,), (0,)))

        return x + (self.alpha / self.rank) * res
    
    def remove_parametrizations(self) -> None:
        pass


class CompositeTransform(BaseTransform):
    
    def __init__(self, transforms: List[BaseTransform]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        for transform in self.transforms:
            x = transform(x, inv_t, dim)
        return x
    
    def remove_parametrizations(self) -> None:
        for transform in self.transforms:
            transform.remove_parametrizations()

class DCTTransform(BaseTransform):
    
    def __init__(self, group_size: int = 128):
        super().__init__()
        import numpy as np
        from scipy.fftpack import dct
        self.group_size = group_size
        self.block_dct = torch.from_numpy(
                dct(np.eye(group_size), type=2, norm='ortho')
                )
        self.mat = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # DCT transform is it own inverse
        x_shape = x.shape
        if self.mat is None:
            self.mat = torch.block_diag(
                *[self.block_dct] * (x_shape[-1] // self.group_size),
            ).to(x.device).to(x.dtype)
        return torch.matmul(x, self.mat)

    def remove_parametrizations(self) -> None:
        pass
    
class DSTransform(BaseTransform):
    
    def __init__(self, group_size: int = 128):
        super().__init__()
        import numpy as np
        from scipy.fftpack import dst
        self.group_size = group_size
        self.block_dct = torch.from_numpy(
                dst(np.eye(group_size), type=2, norm='ortho')
                )
        self.mat = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # DST transform is it own inverse
        x_shape = x.shape
        if self.mat is None:
            self.mat = torch.block_diag(
                *[self.block_dct] * (x_shape[-1] // self.group_size),
            ).to(x.device).to(x.dtype)
        return torch.matmul(x, self.mat)
    
    def remove_parametrizations(self) -> None:
        pass


class FastFoodTransform(BaseTransform):
    def __init__(self, group_size: int = 128):
        '''
        Implemented based on the FastFood transform paper:
            - https://arxiv.org/pdf/1408.3060
            - https://scikit-learn-extra.readthedocs.io/en/stable/_modules/sklearn_extra/kernel_approximation/_fastfood.html#Fastfood
        '''
        
        super().__init__()
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
        sigma = 1 / math.sqrt(2)
        B = torch.diag(torch.randint(0, 2, (group_size,)).float() * 2 - 1)
        G = torch.diag(torch.randn(group_size))
        H = hadamard_transform(torch.eye(group_size, device=device)).cpu()
        S = torch.diag((1 / l2norm_along_axis1(G)) *  sample_chi(group_size))
        P = torch.eye(group_size)[torch.randperm(group_size, device='cpu')].to(torch.float32)
        self.block_mat = (1 / sigma) *(1 / math.sqrt(group_size) )*(S@H@G@P@H@B)
        self.block_inv_mat = torch.linalg.inv(self.block_mat)
        self.group_size = group_size
        self.mat = None
        self.inv_mat = None
        
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if inv_t:
            x_shape = x.shape
            if self.inv_mat is None:
                self.inv_mat = torch.block_diag(
                    *[self.block_inv_mat] * (x_shape[-1] // self.group_size),
                ).to(x.device).to(x.dtype).T
                del self.block_inv_mat
                
            return torch.matmul(x, self.inv_mat)
        else:
            x_shape = x.shape
            if self.mat is None:
                self.mat = torch.block_diag(
                    *[self.block_mat] * (x_shape[-1] // self.group_size),
                ).to(x.device).to(x.dtype)
                del self.block_mat
            return torch.matmul(x, self.mat)


class GSRTransform(BaseTransform):
    
    def __init__(self, group_size: int = 128):
        super().__init__()
        from scipy.linalg import hadamard
        self.group_size = group_size
        q_ = torch.tensor(hadamard(self.group_size), dtype=torch.float64)
        sign_changes = torch.diff(q_, dim=0).ne(0).sum(dim=0) 
        sorted_indices = torch.argsort(sign_changes)
        q_ = q_[:, sorted_indices]
        self.block_gsr= q_ / torch.tensor(q_.shape[-1]).sqrt()

        self.mat = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        # GSR transform is it own inverse
        x_shape = x.shape
        if self.mat is None:
            self.mat = torch.block_diag(
                *[self.block_gsr] * (x_shape[-1] // self.group_size),
            ).to(x.device).to(x.dtype)
        return torch.matmul(x, self.mat)

    def remove_parametrizations(self) -> None:
        pass
        

TRANSFORMS = {
    "identity": IdentityTransform,
    "full": FullTransform,
    "hadamard": HadamardTransform,
    "givens": GivensTransform,
    "kronecker": KroneckerFactorizedTransform,
    "identity_low_rank": IdentityLowRankTransform,
    "dct": DCTTransform,
    "dst": DSTransform,
    "fast_food": FastFoodTransform,
    "gsr": GSRTransform
}


def build_transform(transform_class: str, **transform_kwargs) -> BaseTransform:
    transform = TRANSFORMS[transform_class]
    return transform(**filter_kwarg_dict(transform.__init__, transform_kwargs))

def get_transform_matrix(
    transform_class: str, 
    size: int, 
    device: torch.device = None, 
    dtype: torch.dtype = None
) -> torch.Tensor:
    if transform_class == "hadamard":
        return hadamard_transform(torch.eye(size, device=device, dtype=dtype), scale=1 / math.sqrt(size))
    elif transform_class == "identity":
        return torch.eye(size, device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"get_transform_matrix is implemented only for Hadamard and Identity transforms")
