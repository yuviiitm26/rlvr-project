import argparse
import torch
import re
from unsloth import FastLanguageModel
from sandbox_grader import PythonJailGrader
from data_loader import get_mbpp_80_20

def extract_code(response: str) -> str:
    """Extracts Python code from the LLM's response, ignoring <think> blocks."""
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match: return match.group(1).strip()
    think_stripped = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    return think_stripped.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="grpo_saved_lora", help="Path to saved QLoRA adapters")
    args = parser.parse_args()

    print("="*60)
    print("RLVR Post-Training: Zero-Shot Unseen Evaluation")
    print("="*60)

    # 1. Load Eval Dataset
    _, eval_problems = get_mbpp_80_20()
    grader = PythonJailGrader(use_sandbox=True)
    
    # 2. Load Model + Trained Adapters
    print(f"Loading Base Qwen2.5 and applying adapters from {args.adapter_path}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            max_seq_length=4096,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        model.load_adapter(args.adapter_path)
    except Exception as e:
        print(f"Failed to load model/adapters. Ensure '{args.adapter_path}' exists. Error: {e}")
        return

    # Enable Unsloth native 2x faster inference
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Base models lack a chat_template. Inject a standard ChatML template.
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )
    
    passed = 0
    total = len(eval_problems)
    
    # 3. Strict Zero-Shot Loop (No Self-Correction)
    for i, prob in enumerate(eval_problems):
        tests_str = "\n".join(prob["test_list"][:2])
        prompt = (
            "You are an expert Python programmer. \n"
            "You must first analyze the problem step-by-step inside <think> tags. \n"
            "Then, output your final working code inside a ```python ``` block.\n"
            f"Problem: {prob['description']}\n\n"
            "Your code must pass these example assertions (ensure your function name matches):\n"
            f"{tests_str}"
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        # Enforce memory safety
        with torch.no_grad():
            out_tokens = model.generate(
                **inputs, 
                max_new_tokens=768, 
                temperature=0.1,  # Low temperature for strict coding evaluation
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_code(response)
        
        result = grader.grade(code, prob["test_code"])
        
        if result.passed:
            passed += 1
            print(f"[{i+1:02d}/{total}] {prob['id']} | PASSED")
        else:
            print(f"[{i+1:02d}/{total}] {prob['id']} | FAILED | Reason: {result.error_type}")
            
    # 4. Telemetry Output
    pass_rate = (passed / total) * 100
    print("="*60)
    print(f"Strict Zero-Shot Evaluation Complete")
    print(f"Final Pass Rate: {pass_rate:.2f}% ({passed}/{total} Unseen Problems)")
    print("="*60)

if __name__ == "__main__":
    main()
