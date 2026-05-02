# Abliterix — small vLLM compatibility shims
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility helpers for model families that vLLM supports before HF does."""

from __future__ import annotations

from typing import Any


def install_gemma4_transformers_compat() -> None:
    """Register minimal Gemma4 configs and patch a tokenizer_config quirk.

    vLLM 0.19 includes Gemma4 model executors, but the paired Transformers
    release may not yet know ``model_type="gemma4"``. Gemma4 tokenizer configs
    can also carry ``extra_special_tokens`` as a list, while Transformers 4.x
    expects a mapping. Both fixes are process-local and only affect this run.
    """

    from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

    class Gemma4TextConfig(PretrainedConfig):
        model_type = "gemma4_text"

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)

    class Gemma4Config(PretrainedConfig):
        model_type = "gemma4"

        def __init__(self, text_config: Any | None = None, **kwargs: Any):
            super().__init__(**kwargs)
            if isinstance(text_config, dict):
                self.text_config = Gemma4TextConfig(**text_config)
            elif text_config is None:
                self.text_config = Gemma4TextConfig()
            else:
                self.text_config = text_config

    try:
        AutoConfig.register("gemma4", Gemma4Config, exist_ok=True)
        AutoConfig.register("gemma4_text", Gemma4TextConfig, exist_ok=True)
    except ValueError:
        pass

    try:
        from vllm.transformers_utils import config as vllm_config

        vllm_config._CONFIG_REGISTRY.setdefault("gemma4", Gemma4Config)
        vllm_config._CONFIG_REGISTRY.setdefault("gemma4_text", Gemma4TextConfig)
    except Exception:
        pass

    if getattr(AutoTokenizer.from_pretrained, "_abliterix_gemma4_patch", False):
        return

    original_from_pretrained = AutoTokenizer.from_pretrained

    def from_pretrained_with_gemma4_extra_tokens(*args: Any, **kwargs: Any):
        try:
            return original_from_pretrained(*args, **kwargs)
        except AttributeError as exc:
            if "'list' object has no attribute 'keys'" not in str(exc):
                raise
            kwargs = dict(kwargs)
            kwargs["extra_special_tokens"] = {}
            return original_from_pretrained(*args, **kwargs)

    from_pretrained_with_gemma4_extra_tokens._abliterix_gemma4_patch = True  # type: ignore[attr-defined]
    AutoTokenizer.from_pretrained = from_pretrained_with_gemma4_extra_tokens
