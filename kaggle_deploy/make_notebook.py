import json
import base64

files = [
    "sandbox_grader.py", "rewards.py", "problems.py", "mdp.py", 
    "hf_llm.py", "grpo_trainer.py", "train_rlvr.py",
    "radix_cache.py", "fused_rmsnorm.py", "verify_kernel.py",
    "patch_unsloth.py", "data_loader.py"
]
bundle = {}
for f in files:
    with open("../src/" + f, "rb") as fd:
        bundle[f] = base64.b64encode(fd.read()).decode("utf-8")

setup_code = "import base64\nbundle = " + json.dumps(bundle) + "\n"
setup_code += """
for f, b64 in bundle.items():
    with open(f, "wb") as fd:
        fd.write(base64.b64decode(b64))
print("Files written successfully:")
for f in bundle.keys():
    print(f" - {f}")
"""

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install Unsloth and Triton dependencies\n",
                "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                "!pip install --no-deps xformers trl peft accelerate bitsandbytes datasets\n",
                "!pip install -q triton\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [setup_code]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Phase 3: Systems Optimization Verification\n",
                "print('=== Running Phase 3 Triton Fused RMSNorm Verification ===')\n",
                "!python verify_kernel.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Run the Multi-Turn RLVR Training Loop (MBPP + Masking)\n",
                "!useradd -M -s /bin/false sandboxuser 2>/dev/null || echo 'User already exists'\n",
                "!mkdir -p /tmp/ai_workspace\n",
                "!python train_rlvr.py\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("rlvr_phase2.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Updated rlvr_phase2.ipynb with train_rlvr.py and MBPP.")
