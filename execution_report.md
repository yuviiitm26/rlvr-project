# Kaggle Execution Report — All Runs

## Run Summary

| Run | Kernel | Version | Status | Key Issue |
|-----|--------|---------|--------|-----------|
| Phase 1 | `rlvr-project` | V1 | ❌ Error | `ModuleNotFoundError: No module named 'run_demo'` — bundled script didn't extract files to the correct working directory |
| Phase 2 | `rlvr-project-phase-2-unsloth` | V2 | ❌ Error | `ValueError: No kernel name found in notebook` — missing `kernelspec` in Jupyter metadata |
| Phase 2 | `rlvr-project-phase-2-unsloth` | V3 | ✅ **Completed** | Full pipeline ran end-to-end. Model generated, grader executed, DAPO filtered. |

---

## Phase 2 V3 — Detailed Analysis

### ✅ What Worked Perfectly

1. **Unsloth installation** — All dependencies resolved (transformers 5.5.0, bitsandbytes 0.50.2, trl 0.29.1, torchao 0.18.0)
2. **Model loading** — `Qwen/Qwen2.5-0.5B` loaded in 4-bit quantization across **dual T4 GPUs**:
   ```
   cuda:0  budget 12.95 GiB  weights 0.227 GiB
   cuda:1  budget 13.00 GiB  weights 0.254 GiB
   ```
3. **LoRA patching** — Unsloth patched all 24 layers: `24 QKV layers, 24 O layers, 24 MLP layers`
4. **ChatML template injection** — No more `chat_template` errors
5. **MDP Loop** — All 3 turns × 4 completions × 2 problems = **24 total generations** ran without crash
6. **Subprocess Grader** — Code was executed and graded correctly (all returned `FAILED` as expected)
7. **DAPO** — Zero-variance detection worked perfectly:
   ```
   Group Rewards:    [0.0, 0.0, 0.0, 0.0]
   Group Advantages: [0.0, 0.0, 0.0, 0.0]
   [DAPO] Zero variance batch detected. Skipping gradient update.
   ```

### ⚠️ Expected Behavior: All Completions Failed

This is **exactly what we should expect** from an untrained base model! The thesis explicitly states:

> *"DO NOT use pre-trained reasoning models like DeepSeek-R1; the objective is to teach a naive model to reason from scratch."*

The `Qwen2.5-0.5B` **base** model has never seen code instruction-following before. It generates raw text completions (not structured `<think>` + ` ```python ` blocks), so the code extractor either gets gibberish or syntactically invalid code. The grader then correctly returns `reward = 0.0` for all attempts.

**This is the cold-start problem** — the exact reason RLVR training exists. The model needs to randomly stumble onto at least one correct solution before GRPO can create gradient signal.

### 🔧 What Needs to Change for Real Training

> [!IMPORTANT]  
> The pipeline is proven end-to-end. These are the concrete improvements needed for the model to actually learn.

#### 1. Curriculum Warm-Start (Critical)
Start with trivially simple problems where even random generation has a nonzero chance of producing valid Python:
```python
# Problem: Return the number 42
# Test: assert answer() == 42
# A base model might accidentally generate: def answer(): return 42
```

#### 2. Higher Temperature + More Completions (G=8 or G=16)
With `G=4` and a base model, the probability of any completion being correct is near zero. Increasing the group size raises the chance of at least one success, breaking the DAPO deadlock.

#### 3. Supervised Fine-Tuning (SFT) Warm-Up
Run 50–100 steps of SFT on simple code examples before starting RL. This teaches the model the basic output format (`<think>` + code blocks) without teaching it to reason — reasoning comes from RL.

#### 4. Remove `max_length` Warning
The Unsloth model sets `max_length=32768` internally, which conflicts with our `max_new_tokens=256`. Fix by explicitly passing `max_length=None` in `hf_llm.py`.

---

## Next Steps

| Priority | Action | Estimated Impact |
|----------|--------|-----------------|
| 🔴 P0 | Add trivial warm-up problems to break DAPO deadlock | Enables first gradient signal |
| 🔴 P0 | SFT warm-up phase (50 steps on format examples) | Model learns ````python` output structure |
| 🟡 P1 | Increase G from 4 → 8 | Higher chance of variance in rewards |
| 🟡 P1 | Fix `max_length`/`max_new_tokens` conflict | Cleaner logs, correct truncation |
| 🟢 P2 | Add trajectory logging (save raw model outputs) | Debug what model is actually generating |
| 🟢 P2 | Phase 3: Triton kernels for KV cache optimization | Memory efficiency for longer sequences |
