import hashlib
import os
from functools import lru_cache
from typing import Dict, Optional

import requests
import torch
import yaml
from filelock import FileLock
from tqdm.auto import tqdm

MODEL_PATHS = {
    "transmitter": "transmitter.pt",
    "decoder": "vector_decoder.pt",
    "text300M": "text_cond.pt",
    "image300M": "image_cond.pt",
}

CONFIG_PATHS = {
    "transmitter": "transmitter_config.yaml",
    "decoder": "vector_decoder_config.yaml",
    "text300M": "text_cond_config.yaml",
    "image300M": "image_cond_config.yaml",
    "diffusion": "diffusion_config.yaml",
}

@lru_cache()
def default_cache_dir() -> str:
    return os.path.join(os.path.abspath(os.getcwd()), "model_cache")


def fetch_file_cached(
    url: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, url)
    if os.path.exists(local_path):
        return local_path

def load_config(
    config_name: str,
    progress: bool = False,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
):
    if config_name not in CONFIG_PATHS:
        raise ValueError(
            f"Unknown config name {config_name}. Known names are: {CONFIG_PATHS.keys()}."
        )
    path = fetch_file_cached(
        CONFIG_PATHS[config_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(
    checkpoint_name: str,
    device: torch.device,
    progress: bool = True,
    cache_dir: Optional[str] = None,
    chunk_size: int = 4096,
) -> Dict[str, torch.Tensor]:
    if checkpoint_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown checkpoint name {checkpoint_name}. Known names are: {MODEL_PATHS.keys()}."
        )
    path = fetch_file_cached(
        MODEL_PATHS[checkpoint_name], progress=progress, cache_dir=cache_dir, chunk_size=chunk_size
    )
    return torch.load(path, map_location=device)


def load_model(
    model_name: str,
    device: torch.device,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    from .configs import model_from_config

    model = model_from_config(load_config(model_name, **kwargs), device=device)
    model.load_state_dict(load_checkpoint(model_name, device=device, **kwargs))
    model.eval()
    return model
