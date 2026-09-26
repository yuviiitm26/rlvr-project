import json
import base64
import os

def create_notebook():
    # Read files
    with open("../src/sandbox_grader.py", "r", encoding="utf-8") as f:
        grader_code = f.read()
    with open("../src/problems.py", "r", encoding="utf-8") as f:
        problems_code = f.read()
    with open("../src/eval_rlvr.py", "r", encoding="utf-8") as f:
        eval_code = f.read()

    bundle = {
        "sandbox_grader.py": base64.b64encode(grader_code.encode("utf-8")).decode("utf-8"),
        "problems.py": base64.b64encode(problems_code.encode("utf-8")).decode("utf-8"),
        "eval_rlvr.py": base64.b64encode(eval_code.encode("utf-8")).decode("utf-8")
    }

    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install Unsloth dependencies\n",
                "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                "!pip install --no-deps xformers trl peft accelerate bitsandbytes datasets"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64\n",
                f"bundle = {json.dumps(bundle)}\n",
                "for filename, b64_content in bundle.items():\n",
                "    with open(filename, 'w', encoding='utf-8') as f:\n",
                "        f.write(base64.b64decode(b64_content).decode('utf-8'))\n",
                "print('Files written successfully!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python eval_rlvr.py"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("rlvr_eval.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("Created rlvr_eval.ipynb")

if __name__ == "__main__":
    create_notebook()
