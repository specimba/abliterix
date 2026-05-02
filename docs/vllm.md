[← back to README](../README.md)

# vLLM Runbook

This note records the production lessons from the Gemma 4 31B trial-40 run.
The default posture for large models should be **vLLM-first**. Do not go back
to HuggingFace generation/evaluation for trial loops unless there is a specific
backend blocker.

## Rule Of Thumb

Use vLLM for:

- baseline generation and refusal counting
- every optimization trial
- LLM-judge evaluation batches
- replaying candidate trials
- long prompt sets where HF pipeline parallelism would serialize work

Use HuggingFace only for:

- one-time hidden-state extraction when vLLM/speculators hidden states are not
  available for the model architecture
- final safetensors export after the winning trial is selected
- quick debugging of model structure or tokenizer issues

For Gemma 4 31B, vLLM hidden-state extraction is not yet reliable, so Phase 1
may still load HF once to compute steering vectors. That is acceptable. The
slow path we must avoid is HF `generate()` for the optimization/eval loop.

## Gemma 4 31B Known-Good Config

The winning run used `configs/gemma4_31b.toml` with these critical settings:

```toml
[model]
backend = "vllm"
tensor_parallel_size = 1
gpu_memory_utilization = 0.92
enable_expert_parallel = false
enforce_eager = true
max_model_len = 4096
disable_lora = true
use_in_place_editing = true

[inference]
max_batch_size = 8
max_gen_tokens = 150
min_gen_tokens = 100

[steering]
steering_mode = "direct"
weight_normalization = "pre"
disabled_components = ["mlp.down_proj"]
strength_range = [1.0, 6.0]

[detection]
llm_judge = true
llm_judge_model = "google/gemini-3-flash-preview"
llm_judge_batch_size = 10
llm_judge_concurrency = 35
```

Why these matter:

- `backend = "vllm"` keeps generation and scoring on the fast backend.
- `disable_lora = true` avoids the Gemma 4 vLLM LoRA path, which produced
  identical base/adapted outputs in testing.
- `use_in_place_editing = true` applies direct projection to the vLLM resident
  model instead of serializing adapters per trial.
- `disabled_components = ["mlp.down_proj"]` keeps the selected Gemma 4 31B run
  on attention projections only; this was more stable than editing MLP down
  projections.
- `min_gen_tokens = 100` is required for honest refusal detection because Gemma
  4 often delays refusals until after a long preamble.

## Launch Template

Use the training venv and local `.env`; do not paste tokens into the shell.

```bash
cd /workspace/abliterix
set -a && source .env && set +a

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

nohup /workspace/venvs/abliterix-vllm/bin/abliterix \
  --config configs/gemma4_31b.toml \
  --non-interactive \
  --overwrite-checkpoint \
  --optimization.checkpoint-dir=/workspace/checkpoints_gemma4_31b_vllm_inplace \
  > /workspace/logs/gemma4_31b_vllm_inplace.log 2>&1 &
echo $! > /workspace/logs/gemma4_31b_vllm_inplace.pid
```

Progress check:

```bash
pid=$(cat /workspace/logs/gemma4_31b_vllm_inplace.pid)
ps -p "$pid" -o pid,stat,etime,pcpu,pmem,rss,cmd
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
grep -aE "Running trial|LLM judge:|Refusals:|Estimated remaining|Optimization complete|Traceback|RuntimeError" \
  /workspace/logs/gemma4_31b_vllm_inplace.log | tail -120
```

## Results From The Reference Run

Hardware:

- RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM
- vLLM backend, in-place direct attention editing
- Gemini LLM judge: `google/gemini-3-flash-preview`

Outcome:

- baseline: `99/100` refusals
- best trial: `trial 40`
- best score: `7/100` refusals
- top safe over-refusal probe: `0/15` refusals for trials 40, 46, and 53
- full 60-trial run time: about 1.5 hours after setup

The judge was not the bottleneck. Gemini judge batching with concurrency 35 was
fast enough; the bottleneck was model generation and initial model/vector setup.

## Pitfalls We Already Hit

### Do Not Use The Gemma 4 vLLM LoRA Path

For Gemma 4 31B, the vLLM LoRA adapter route was effectively inert in testing:
base and adapted outputs were identical and logprob differences were zero. Use
`disable_lora = true` plus `use_in_place_editing = true`.

### Do Not Enable Tower-Connector LoRA For Gemma 4 Text Runs

Trying `enable_tower_connector_lora=True` crashed under the text-only Gemma 4
setup with `ValueError: max() arg is an empty sequence` around
`limit_mm_per_prompt`. Keep the text-only vLLM path clean.

### KL Uses Continuation NLL In vLLM In-Place Mode

In-place editing can make vLLM sampler logprobs look exactly `0.0000` even when
refusal counts clearly change. The vLLM backend therefore scores the baseline
benign continuations with a fresh prefill pass and reports mean continuation NLL
drift as the KL-like damage metric for in-place runs. Treat refusal counts and
replay tests as primary, but this metric should now move when Gemma-style
in-place edits actually perturb benign behavior.

### Prefix Cache Must Not Hide Weight Edits

The vLLM backend disables prefix caching when in-place editors are active. If
that changes, every edit must reset the prefix cache before the next generation,
or stale KV/logprob data can make trials look unchanged.

### Blackwell PCIe Needs Conservative vLLM Settings

On RTX PRO 6000 Blackwell PCIe, keep:

- `enforce_eager = true` for reliability during in-place edits
- custom all-reduce disabled in the vLLM backend
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`

These settings trade a little throughput for predictable worker startup and
weight-edit behavior.

## Export And Publish

After selecting a winning trial, export once with HF so the Hugging Face repo
contains normal safetensors:

```bash
/workspace/venvs/abliterix-vllm/bin/python scripts/export_model.py \
  --model google/gemma-4-31B-it \
  --checkpoint /workspace/checkpoints_gemma4_31b_vllm_inplace \
  --trial 40 \
  --config configs/gemma4_31b.toml \
  --save-local /workspace/export_gemma4_31b_trial40
```

Do not use the generic `upload_model.py` model card for Gemma 4 releases. Write
the README explicitly from the prior model card style and include:

- selected trial
- baseline refusal count
- eval refusal count
- generation length
- judge model
- safe over-refusal probe JSON
- continuation-NLL KL mode for vLLM in-place runs
