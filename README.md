# RLVR Framework - Cloud-Native Reinforcement Learning with Verifiable Rewards

## Overview
Infrastructure for training small language models (SLMs) to reason and self-correct using compiler/test tracebacks via Reinforcement Learning with Verifiable Rewards (RLVR).

**Core Principle:** The compiler is the reward function. No human-in-the-loop.

## Project Structure
```
rlvr-framework/
├── grader.py       # Subprocess execution engine (the reward function)
├── rewards.py      # Reward computation (binary, discounted, GRPO, DAPO)
├── mdp.py          # Multi-turn MDP orchestrator
├── fake_llm.py     # Deterministic mock LLM for testing
├── problems.py     # Problem bank with test assertions
├── run_demo.py     # End-to-end validation demo
└── README.md       # This file
```

## Quick Start

### Local Development (Windows/Mac/Linux)
```bash
# No sandbox needed for local testing
python run_demo.py --no-sandbox
```

### Cloud (Kaggle/Colab with sandbox)
```bash
# After running the sandbox initialization from Step 1
python3 run_demo.py
```

## Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Problem Bank │────►│  MDP Loop    │────►│   LLM / FakeLLM  │
│  (problems.py)│     │  (mdp.py)    │◄────│   (fake_llm.py)  │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │
                     ┌──────▼───────┐     ┌──────────────────┐
                     │   Grader     │────►│  Reward Functions │
                     │  (grader.py) │     │  (rewards.py)     │
                     └──────────────┘     └──────────────────┘
```

## Dependencies
- Python 3.8+
- NumPy (for GRPO advantage computation)

## Status
- [x] Step 1: Cloud sandbox initialization
- [x] Step 2: Subprocess grader + Fake LLM harness
- [ ] Step 3: HuggingFace model integration
- [ ] Step 4: Unsloth QLoRA GRPO training loop

## AI Agent Instructions
- **GitHub Pushing:** Do NOT automatically push every minor fix or iteration to GitHub. Only execute the github_push.py script or push code to the remote repository when the user explicitly requests it.
