"""vec2text compatibility layer (offline, transformers 4.x environment).

Why this module exists
----------------------
* ``vec2text`` 0.0.13 is incompatible with transformers 5.x: its model classes
  call ``from_pretrained`` inside another ``from_pretrained`` (nested model
  construction), and transformers 5 wraps construction in a meta-device context,
  which raises "You are using from_pretrained with a meta device context
  manager".  The reproduction therefore provides a secondary environment
  ``.venv-vec2text`` with transformers 4.57.6.
* All model weights used here already exist in the local HuggingFace cache
  (downloaded previously).  We load them from the cache snapshots with
  ``local_files_only=True`` so the experiment is:
    - reproducible offline,
    - independent from flaky model-hub downloads in this network,
    - identical to the reference vec2text usage (same repos/weights).
  Locate the cache with ``CAPRISE_HF_CACHE`` (default ``~/.cache/huggingface``).

Usage
-----
    from scripts.vec2text_compat import load_local_corrector
    corrector = load_local_corrector()                    # gtr-base corrector
    texts = vec2text.invert_embeddings(embeddings, corrector=corrector, num_steps=20)

Run under ``.venv-vec2text/bin/python``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["snapshot_path", "patch_local_models", "load_local_corrector"]

_HF_CACHE = Path(os.environ.get("CAPRISE_HF_CACHE", Path.home() / ".cache/huggingface"))
_REPOS = {
    "t5-base": "models--t5-base",
    "gtr_embedder": "models--sentence-transformers--gtr-t5-base",
    "inversion": "models--jxm--gtr__nq__32",
    "corrector": "models--jxm--gtr__nq__32__correct",
}

_PATCHED = False


def snapshot_path(repo: str) -> str:
    """Return the newest local snapshot directory for a known HF repo."""
    if repo not in _REPOS:
        raise KeyError(f"unknown repo alias {repo!r}; known: {sorted(_REPOS)}")
    root = _HF_CACHE / "hub" / _REPOS[repo] / "snapshots"
    snaps = sorted(p for p in root.glob("*/") if p.is_dir())
    if not snaps:
        raise FileNotFoundError(
            f"no local snapshot for {_REPOS[repo]} under {root}; "
            f"set CAPRISE_HF_CACHE to the HF cache root"
        )
    return str(snaps[-1])


def patch_local_models() -> None:
    """Redirect transformers/vec2text model lookups to the local snapshots.

    * ``t5-base`` (base weights used to initialise the vec2text encoder-decoder)
    * ``sentence-transformers/gtr-t5-base`` (the embedder used by vec2text)
    * vec2text's ``load_embedder_and_tokenizer("gtr_base")``
    """
    global _PATCHED
    if _PATCHED:
        return

    import transformers

    t5_path = snapshot_path("t5-base")
    gtr_path = snapshot_path("gtr_embedder")
    mapping = {
        "t5-base": t5_path,
        "sentence-transformers/gtr-t5-base": gtr_path,
    }

    def _make_wrapper(orig):
        def wrapper(name_or_path, *args, **kwargs):
            name_or_path = mapping.get(str(name_or_path), name_or_path)
            kwargs.setdefault("local_files_only", True)
            return orig(name_or_path, *args, **kwargs)

        return wrapper

    for cls_name in ("AutoModelForSeq2SeqLM", "AutoModel", "AutoTokenizer", "AutoConfig"):
        cls = getattr(transformers, cls_name)
        cls.from_pretrained = _make_wrapper(cls.from_pretrained)

    # vec2text loads the gtr embedder through load_embedder_and_tokenizer
    import vec2text.models.corrector_encoder as cor_mod
    import vec2text.models.inversion as inv_mod
    import vec2text.models.model_utils as mu

    _orig_loader = mu.load_embedder_and_tokenizer

    def _patched_loader(name, torch_dtype="float32", **kwargs):
        if name == "gtr_base":
            model = transformers.AutoModel.from_pretrained(gtr_path, local_files_only=True).encoder
            tokenizer = transformers.AutoTokenizer.from_pretrained(gtr_path, local_files_only=True)
            return model, tokenizer
        return _orig_loader(name, torch_dtype=torch_dtype, **kwargs)

    for mod in (mu, inv_mod, cor_mod):
        if hasattr(mod, "load_embedder_and_tokenizer"):
            mod.load_embedder_and_tokenizer = _patched_loader

    _PATCHED = True


def load_local_corrector():
    """Load the reference vec2text gtr-base corrector from local snapshots."""
    patch_local_models()
    import vec2text
    import vec2text.models as vm

    inversion = vm.InversionModel.from_pretrained(snapshot_path("inversion"), local_files_only=True)
    corrector_model = vm.CorrectorEncoderModel.from_pretrained(snapshot_path("corrector"), local_files_only=True)
    return vec2text.load_corrector(inversion, corrector_model)
