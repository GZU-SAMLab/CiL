"""
CBSE: Combinatorial Block Sparse Encoding
Utilities for generating deterministic sparse anchors that can be shared
between DER and non-DER training code paths.
"""

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ImportanceMasksInput = Optional[
    Union[
        torch.Tensor,
        np.ndarray,
        Dict[int, Sequence[float]],
        Sequence[Sequence[float]],
    ]
]


def get_dataset_config(dataset_name: str, num_classes: int) -> Dict[str, int]:
    """
    Return a recommended CBSE configuration for the given dataset.
    """
    if num_classes == 10:  # CIFAR-10
        return {"num_blocks": 10, "blocks_per_class": 1}
    if num_classes == 100:  # CIFAR-100 / ImageNet-100
        return {"num_blocks": 15, "blocks_per_class": 2}
    if num_classes == 200:  # CUB-200
        return {"num_blocks": 12, "blocks_per_class": 3}
    return auto_config(num_classes)


def auto_config(num_classes: int, total_dim: int = 512) -> Dict[str, int]:
    """
    Automatically find a CBSE configuration when a dataset-specific one is
    not available.
    """
    from scipy.special import comb as scipy_comb

    best_config = None
    min_waste = float("inf")

    for blocks_per_class in [1, 2, 3, 4]:
        for num_blocks in range(blocks_per_class, 50):
            try:
                num_combinations = scipy_comb(num_blocks, blocks_per_class, exact=True)
            except Exception:
                num_combinations = comb_manual(num_blocks, blocks_per_class)

            if num_combinations < num_classes:
                continue

            block_size = total_dim / num_blocks
            if not 25 <= block_size <= 60:
                continue

            waste_rate = (num_combinations - num_classes) / num_classes
            if waste_rate < 0.3 and waste_rate < min_waste:
                min_waste = waste_rate
                best_config = {
                    "num_blocks": num_blocks,
                    "blocks_per_class": blocks_per_class,
                }
                break

    if best_config is None:
        if num_classes <= 20:
            best_config = {"num_blocks": num_classes, "blocks_per_class": 1}
        else:
            best_config = {"num_blocks": 15, "blocks_per_class": 2}

    return best_config


def comb_manual(n: int, k: int) -> int:
    """Compute C(n, k) manually."""
    if k > n or k < 0:
        return 0
    if k in (0, n):
        return 1

    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _sanitize_mask_vector(
    mask: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]]
) -> Optional[torch.Tensor]:
    """
    Convert any supported mask representation into a 1D float tensor.
    """
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        vec = mask.detach().float()
    elif isinstance(mask, np.ndarray):
        vec = torch.from_numpy(mask).float()
    else:
        vec = torch.tensor(mask, dtype=torch.float32)

    vec = vec.flatten()
    if vec.numel() == 0:
        return None
    return vec


def _prepare_importance_masks(
    importance_masks: ImportanceMasksInput, num_classes: int
) -> Optional[List[Optional[torch.Tensor]]]:
    """
    Normalize different mask input formats into a list of tensors indexed by class id.
    """
    if importance_masks is None:
        return None

    mask_list: List[Optional[torch.Tensor]] = [None] * num_classes

    if isinstance(importance_masks, torch.Tensor):
        if importance_masks.dim() != 2 or importance_masks.size(0) != num_classes:
            raise ValueError(
                "[CBSE] importance_masks tensor must have shape [num_classes, dim]."
            )
        for idx in range(num_classes):
            mask_list[idx] = _sanitize_mask_vector(importance_masks[idx])
        return mask_list

    if isinstance(importance_masks, np.ndarray):
        if importance_masks.ndim != 2 or importance_masks.shape[0] != num_classes:
            raise ValueError(
                "[CBSE] importance_masks ndarray must have shape [num_classes, dim]."
            )
        for idx in range(num_classes):
            mask_list[idx] = _sanitize_mask_vector(importance_masks[idx])
        return mask_list

    if isinstance(importance_masks, dict):
        for class_id, mask in importance_masks.items():
            if 0 <= class_id < num_classes:
                mask_list[class_id] = _sanitize_mask_vector(mask)
        return mask_list

    if isinstance(importance_masks, Sequence):
        if len(importance_masks) != num_classes:
            raise ValueError(
                "[CBSE] importance_masks sequence length must equal num_classes."
            )
        for idx, mask in enumerate(importance_masks):
            mask_list[idx] = _sanitize_mask_vector(mask)
        return mask_list

    raise TypeError("[CBSE] Unsupported importance_masks type.")


def _compute_similarity_matrix(
    mask_list: List[Optional[torch.Tensor]]
) -> Optional[torch.Tensor]:
    """
    Compute a cosine-similarity matrix from the provided mask list.
    """
    valid_entries = [
        (idx, vec) for idx, vec in enumerate(mask_list) if vec is not None
    ]

    if not valid_entries:
        return None

    first_dim = valid_entries[0][1].numel()
    for _, vec in valid_entries:
        if vec.numel() != first_dim:
            raise ValueError(
                "[CBSE] Importance masks have inconsistent feature dimensions."
            )

    stacked = torch.stack([vec for _, vec in valid_entries], dim=0)
    stacked = stacked / (stacked.norm(dim=1, keepdim=True) + 1e-8)
    similarity_sub = stacked @ stacked.t()

    similarity_full = torch.zeros(
        len(mask_list), len(mask_list), dtype=similarity_sub.dtype
    )
    for row_offset, (row_idx, _) in enumerate(valid_entries):
        for col_offset, (col_idx, _) in enumerate(valid_entries):
            similarity_full[row_idx, col_idx] = similarity_sub[row_offset, col_offset]

    return similarity_full


def generate_block_allocation(
    num_classes: int,
    num_blocks: int,
    blocks_per_class: int,
    importance_masks: ImportanceMasksInput = None,
    confusion_top_k: int = 1,
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """
    Generate a block allocation plan. Optionally avoids block overlap between
    new classes and their most similar (confusing) previously allocated classes.
    """
    if confusion_top_k < 0:
        raise ValueError("[CBSE] confusion_top_k must be non-negative.")

    all_combinations = list(combinations(range(num_blocks), blocks_per_class))
    num_combinations = len(all_combinations)

    if num_combinations < num_classes:
        raise ValueError(
            "[CBSE] Not enough unique block combinations to cover all classes. "
            f"Got {num_combinations} combinations for {num_classes} classes."
        )

    print(
        f"[CBSE] Generating block allocation: {num_blocks} blocks, "
        f"{blocks_per_class} per class"
    )
    print(
        f"[CBSE] Total combinations: {num_combinations}, needed: {num_classes}, "
        f"waste rate: {100 * (num_combinations - num_classes) / num_classes:.1f}%"
    )

    mask_list = (
        _prepare_importance_masks(importance_masks, num_classes)
        if importance_masks is not None
        else None
    )
    similarity_matrix = (
        _compute_similarity_matrix(mask_list) if mask_list is not None else None
    )

    allocation: Dict[int, List[int]] = {}
    allocation_matrix = np.zeros((num_classes, num_blocks), dtype=np.int32)
    used_combo_indices = set()

    for class_id in range(num_classes):
        avoid_blocks = set()
        if (
            similarity_matrix is not None
            and mask_list is not None
            and mask_list[class_id] is not None
            and class_id > 0
        ):
            previous_indices = [
                idx for idx in range(class_id) if mask_list[idx] is not None
            ]
            if previous_indices:
                similarities = torch.tensor(
                    [similarity_matrix[class_id, idx].item() for idx in previous_indices]
                )
                top_k = min(confusion_top_k, len(previous_indices))
                if top_k > 0:
                    _, top_positions = torch.topk(similarities, k=top_k)
                    confused_classes = [
                        previous_indices[pos] for pos in top_positions.tolist()
                    ]
                    for confused in confused_classes:
                        avoid_blocks.update(allocation.get(confused, []))

        chosen_combo_idx: Optional[int] = None
        fallback_combo_idx: Optional[int] = None

        for combo_idx, combo in enumerate(all_combinations):
            if combo_idx in used_combo_indices:
                continue
            if fallback_combo_idx is None:
                fallback_combo_idx = combo_idx
            if avoid_blocks and any(block in avoid_blocks for block in combo):
                continue
            chosen_combo_idx = combo_idx
            break

        if chosen_combo_idx is None:
            chosen_combo_idx = fallback_combo_idx
            if avoid_blocks:
                print(
                    f"[CBSE][Warning] Class {class_id} could not avoid confusing "
                    f"blocks. Falling back to combination {all_combinations[chosen_combo_idx]}."
                )

        used_combo_indices.add(chosen_combo_idx)
        blocks = list(all_combinations[chosen_combo_idx])
        allocation[class_id] = blocks
        allocation_matrix[class_id, blocks] = 1

    print("[CBSE] Block allocation finished.")
    for block_id in range(min(5, num_blocks)):
        count = allocation_matrix[:, block_id].sum()
        print(f"  Block {block_id}: used by {count} classes")
    if num_blocks > 5:
        print("  ...")

    return allocation, allocation_matrix


def generate_sparse_cbse_vectors(
    dim: int,
    num_classes: int,
    block_allocation: Dict[int, List[int]],
    block_allocation_matrix: np.ndarray,
    use_random: bool = True,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build sparse CBSE vectors from a block allocation plan.
    """
    num_blocks = block_allocation_matrix.shape[1]
    block_size = dim // num_blocks

    print(f"[CBSE] Building sparse vectors: dim={dim}, classes={num_classes}")
    print(f"[CBSE] Block size: {block_size}")

    sparse_vectors = torch.zeros(dim, num_classes, dtype=torch.float32)

    for class_id in range(num_classes):
        for block_idx in block_allocation[class_id]:
            start_dim = block_idx * block_size
            end_dim = min((block_idx + 1) * block_size, dim)
            block_dim = end_dim - start_dim

            torch.manual_seed(seed + class_id * 1000 + block_idx)
            block_vec = torch.randn(block_dim)
            if not use_random:
                block_vec = torch.randn(block_dim)

            block_vec = block_vec / (torch.norm(block_vec) + 1e-8)
            sparse_vectors[start_dim:end_dim, class_id] = block_vec

        norm = torch.norm(sparse_vectors[:, class_id])
        if norm > 1e-8:
            sparse_vectors[:, class_id] = sparse_vectors[:, class_id] / norm

    non_zero_ratio = (sparse_vectors != 0).float().mean().item()
    print(f"[CBSE] Sparsity: {non_zero_ratio * 100:.1f}% (non-zero ratio)")
    active_dims = (sparse_vectors != 0).sum(dim=0).float().mean().item()
    print(f"[CBSE] Active dims per class (avg): {active_dims:.0f}")

    return sparse_vectors


def generate_cbse_anchor(
    dim: int,
    num_classes: int,
    dataset_name: str = "",
    seed: int = 42,
    importance_masks: ImportanceMasksInput = None,
    confusion_top_k: int = 1,
) -> torch.Tensor:
    """
    High-level helper that produces CBSE anchors. If importance_masks is provided,
    new classes try to avoid reusing blocks used by their most similar predecessors.
    """
    print(f"\n{'=' * 60}")
    print("[CBSE] Initialising combinatorial block sparse encoding")
    print(f"{'=' * 60}")

    config = get_dataset_config(dataset_name, num_classes)
    num_blocks = config["num_blocks"]
    blocks_per_class = config["blocks_per_class"]

    print(f"[CBSE] Dataset: {dataset_name or 'Auto'}")
    print(f"[CBSE] Classes: {num_classes}, Feature dim: {dim}")
    print(f"[CBSE] Config: {num_blocks} blocks, {blocks_per_class} per class")

    block_allocation, block_allocation_matrix = generate_block_allocation(
        num_classes=num_classes,
        num_blocks=num_blocks,
        blocks_per_class=blocks_per_class,
        importance_masks=importance_masks,
        confusion_top_k=confusion_top_k,
    )

    cbse_vectors = generate_sparse_cbse_vectors(
        dim=dim,
        num_classes=num_classes,
        block_allocation=block_allocation,
        block_allocation_matrix=block_allocation_matrix,
        use_random=True,
        seed=seed,
    )

    print(f"[CBSE] Finished! Anchor shape: {cbse_vectors.shape}")
    print(f"{'=' * 60}\n")
    return cbse_vectors


if __name__ == "__main__":
    print("Testing CBSE generation...\n")

    anchor_cifar10 = generate_cbse_anchor(512, 10, "cifar10")
    print(f"CIFAR-10 anchor shape: {anchor_cifar10.shape}\n")

    anchor_cifar100 = generate_cbse_anchor(512, 100, "cifar100")
    print(f"CIFAR-100 anchor shape: {anchor_cifar100.shape}\n")

    anchor_cub = generate_cbse_anchor(512, 200, "cub")
    print(f"CUB-200 anchor shape: {anchor_cub.shape}\n")
