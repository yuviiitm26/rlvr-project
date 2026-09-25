import json
import base64

files = [
    "sandbox_grader.py", "data_loader.py", "evaluate_rlvr.py"
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
print("Evaluation scripts injected.")
"""

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install dependencies\n",
                "!useradd -M -s /bin/false sandboxuser || true\n",
                "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                "!pip install --no-deps xformers trl peft accelerate bitsandbytes datasets\n"
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
                "!python evaluate_rlvr.py --adapter_path /kaggle/input/rlvr-project-phase-2-unsloth/grpo_saved_lora\n"
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
with open("rlvr_eval.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f)
print("Created rlvr_eval.ipynb")
