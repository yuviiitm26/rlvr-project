# RLVR Post-Training Evaluation Suite

Once the 320 GRPO rollouts (Version 12) have finished executing on the remote Kaggle T4 GPUs, you will need to retrieve the artifacts and run the Evaluation Suite.

## 1. Retrieving Kaggle Outputs
Because Kaggle runs as an isolated ephemeral environment, the training logs and saved LoRA adapter weights must be downloaded to your local workspace.

Navigate to the `Data -> Output` section of the completed Kaggle Version 12 notebook and download:
1. The **`grpo_saved_lora/`** directory (contains `adapter_config.json` and `adapter_model.safetensors`).
2. The raw kernel logs (save this text file as **`training.log`**).

Place both of these in the root `rlvr-framework/` directory.

## 2. Generating Telemetry Graphs
Run the `plot_metrics.py` script against the retrieved `training.log`:
```bash
python plot_metrics.py --log_file training.log
```
This will parse the standard output from the Kaggle execution and generate a high-resolution `telemetry_metrics.png` displaying:
* **Graph 1:** The trajectory of Mean Verifiable Rewards (proving the model learned to write passing code).
* **Graph 2:** The surrogate Policy Loss layered with the KL Divergence penalty (proving the old and new policies did not drastically diverge).

## 3. The Zero-Shot Unseen Harness
Run the evaluation script to test the model's generalized reasoning on the held-out `eval_dataset`:
```bash
python evaluate_rlvr.py --adapter_path ./grpo_saved_lora
```
**Evaluation Constraints:**
* **`torch.no_grad()`** is strictly enforced to prevent memory tracking.
* The script utilizes **Low-Temperature (T=0.1) Sampling** to test deterministic zero-shot capability.
* **No Test-Time Compute**: Unlike training, the `PythonJailGrader` is only invoked once per problem. If the model fails the assertion tests, it does NOT receive the traceback for self-correction. 
* The final output will yield an un-biased, strict zero-shot pass rate (%).
