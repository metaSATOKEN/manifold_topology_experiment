#!/usr/bin/env python3
# Copyright 2026 Kentaro Sato
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Shared hidden-state extraction utilities.

Reuses the extraction logic from geometric_phase_extraction,
adapted for the manifold topology investigation.
"""

import gc
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
)
from sklearn.decomposition import PCA


# ── Models ──────────────────────────────────────────────────────────────────

MODELS = {
    # Post-LN
    "GPT-1": {
        "name": "openai-community/openai-gpt",
        "n_layers": 12,
        "d_model": 768,
        "ln_type": "Post-LN",
        "dtype": None,
        "loader": "openai-gpt",
    },
    # Pre-LN
    "GPT-2": {
        "name": "gpt2",
        "n_layers": 12,
        "d_model": 768,
        "ln_type": "Pre-LN",
        "dtype": None,
        "loader": "auto",
    },
    "OPT-125m": {
        "name": "facebook/opt-125m",
        "n_layers": 12,
        "d_model": 768,
        "ln_type": "Pre-LN",
        "dtype": None,
        "loader": "auto",
    },
    "OPT-1.3B": {
        "name": "facebook/opt-1.3b",
        "n_layers": 24,
        "d_model": 2048,
        "ln_type": "Pre-LN",
        "dtype": None,
        "loader": "auto",
    },
}

SAMPLE_TEXTS = {
    "narrative": (
        "The old lighthouse keeper climbed the spiral staircase each evening, "
        "carrying a lantern that cast long shadows across the stone walls. "
        "He had performed this ritual for forty years, ever since the automated "
        "systems had failed during the great storm. The sea below crashed "
        "against the rocks with a rhythm that matched his breathing, and he "
        "found comfort in the predictability of waves and wind. Tonight, "
        "however, something was different. A strange light flickered on the "
        "horizon, pulsing with an irregular beat that made him uneasy."
    ),
    "technical": (
        "Transformer architectures process sequential input through layers of "
        "multi-head self-attention and feed-forward networks. Each attention "
        "head computes query, key, and value projections, producing a weighted "
        "sum over value vectors. The weights are determined by the softmax of "
        "scaled dot-product similarities between queries and keys. Residual "
        "connections and layer normalization stabilize training across many "
        "layers, enabling models with billions of parameters to converge."
    ),
    "repetitive": (
        "tick tock tick tock tick tock the clock goes round and round "
        "tick tock tick tock tick tock the pendulum swings back and forth "
        "tick tock tick tock tick tock time flows like a river "
        "tick tock tick tock tick tock the seconds become minutes "
        "tick tock tick tock tick tock and minutes become hours "
    ),
}


# ── Extraction ──────────────────────────────────────────────────────────────

def extract_hidden_states(
    model_key: str,
    text: str,
    layer_indices: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Extract hidden-state trajectories from a model.

    Returns:
        {layer_idx: array of shape (T, d_model)}
    """
    mcfg = MODELS[model_key]
    model_name = mcfg["name"]
    dtype = mcfg.get("dtype")
    loader = mcfg.get("loader", "auto")

    if loader == "openai-gpt":
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
        model = OpenAIGPTLMHeadModel.from_pretrained(
            model_name, output_hidden_states=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {"output_hidden_states": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
        )

    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Truncate to model's max position embeddings if needed
    max_len = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", None
    )
    if max_len is not None and input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]

    n_layers = mcfg["n_layers"]

    if layer_indices is None:
        layer_indices = sorted(
            set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1])
        )

    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states

    trajectories = {}
    for li in layer_indices:
        trajectories[li] = hidden_states[li + 1][0].float().cpu().numpy()

    del outputs, hidden_states, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return trajectories


def project_to_pca2(trajectory: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Project trajectory to top-2 PCA components.

    Returns:
        (projected: (T, 2), variance_explained: float)
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(trajectory)
    var_exp = float(pca.explained_variance_ratio_.sum())
    return projected, var_exp


def get_best_layer(
    trajectories: dict[int, np.ndarray],
    n_layers: int,
) -> int:
    """Return the mid-layer index (closest to n_layers // 2)."""
    mid = n_layers // 2
    return min(trajectories.keys(), key=lambda x: abs(x - mid))
