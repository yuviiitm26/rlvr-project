import json

nb = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
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
    "# Clone Repo\n",
    "!rm -rf rlvr-project\n",
    "!git clone https://yuviiitm26:<GITHUB_TOKEN>@github.com/yuviiitm26/rlvr-project.git\n",
    "%cd rlvr-project"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set Permissions and Evaluate\n",
    "!chmod +x src/setup_sandbox.sh\n",
    "!./src/setup_sandbox.sh\n",
    "!python src/eval_rlvr.py"
   ]
  }
 ],
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

with open("kaggle_deploy/rlvr_eval.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
