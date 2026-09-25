# RLVR Sandbox Regex Parsing Strategy

To successfully implement Test-Time Compute via a `<think>` scratchpad, we must safely decouple the model's textual reasoning from the executable code sent to the POSIX Sandbox Grader.

## 1. Enforcing the Structure
We update the GRPO System Prompt inside `train_rlvr.py`:
```text
You are an expert Python programmer. 
You must first analyze the problem step-by-step inside <think> tags. 
Then, output your final working code inside a ```python ``` block.
Problem: {problem_text}
```

## 2. The Extraction Regex (`sandbox_grader.py` / `mdp.py`)
When the model replies, it will generate a sequence like this:
```
<think>
To solve this, I need to iterate over the list and keep a running sum.
</think>
```python
def sum_list(lst):
    return sum(lst)
```
```

The extraction logic utilizes a 3-tier priority fallback system to isolate the code:

### Priority 1: Explicit Python Blocks
```python
match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
```
- `r'```python'` looks for the exact Markdown code block prefix.
- `(.*?)` is a non-greedy capture group that grabs everything up to the first closing sequence.
- `re.DOTALL` ensures that the `.` character matches newlines (since code spans multiple lines).
- **Result:** Extracts exactly what is inside the markdown block, ignoring the preceding `<think>` tags entirely.

### Priority 2: Generic Code Blocks
```python
match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
```
- If the model forgets to write "python" next to the backticks, we fallback to matching generic code blocks.

### Priority 3: Think-Tag Stripping
```python
think_match = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
return think_match.strip()
```
- If the model completely fails to wrap its output in backticks, we use `re.sub` to aggressively delete everything inside `<think>`...`</think>`. Whatever is left over is sent to the compiler.

## 3. Masking the Loss
Because we only want to optimize the model's actual reasoning and output, our updated `train_rlvr.py` logic constructs a PyTorch `labels` tensor where:
- Prompt tokens → `-100` (Ignored in PyTorch `CrossEntropyLoss`)
- Traceback tokens → `-100` (Ignored)
- `<think>` and ````python```` generated tokens → Retain their token IDs (Gradients ON)
