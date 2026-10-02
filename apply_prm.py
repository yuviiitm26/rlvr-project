import re

with open('src/rewards.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add import re if needed
if 'import re' not in content:
    content = content.replace('from typing import List', 'import re\nfrom typing import List')

prm_class = '''
# ────────────────────────────────────────────────────────────────
# Process Reward Model (PRM) Scaffolding
# ────────────────────────────────────────────────────────────────

class HeuristicPRM:
    """
    Scaffolding for a Process Reward Model (PRM) ala OpenAI's o1.
    
    Currently implemented as a Heuristic PRM to evaluate the quality 
    of the intermediate reasoning steps inside <think> tags. 
    This class is designed to be hot-swappable with a Neural PRM (e.g., 
    a HuggingFace sequence classification model) when scaling up compute.
    """
    def __init__(self):
        # Keyword triggers for heuristic evaluation
        self.planning_words = [r"\\bfirst\\b", r"\\bthen\\b", r"\\bfinally\\b", r"\\bstep \\d+\\b", r"^\\s*\\d+\\.\\s+"]
        self.edge_case_words = [r"edge case", r"empty", r"negative", r"zero", r"boundary", r"null", r"none"]
        self.reflection_words = [r"wait", r"actually", r"however", r"incorrect", r"let me", r"re-read", r"rethink", r"no,"]
        self.complexity_words = [r"time complexity", r"o\\(n", r"space complexity", r"efficient", r"optimize"]

    def evaluate(self, think_content: str) -> float:
        """
        Evaluates the reasoning trace and assigns a dense process reward.
        Max process reward = 0.5
        """
        if not think_content or not think_content.strip():
            return 0.0

        think_lower = think_content.lower()
        prm_score = 0.0

        # 1. Structure (0.1): Did it break thoughts into multiple paragraphs?
        paragraphs = [p for p in think_content.split("\\n\\n") if p.strip()]
        if len(paragraphs) >= 3:
            prm_score += 0.1

        # 2. Planning (0.1): Did it formulate a step-by-step plan?
        if any(re.search(pattern, think_lower, re.MULTILINE) for pattern in self.planning_words):
            prm_score += 0.1

        # 3. Edge Cases (0.1): Did it consider boundary conditions?
        if any(re.search(pattern, think_lower) for pattern in self.edge_case_words):
            prm_score += 0.1

        # 4. Reflection (0.1): Did it exhibit self-correction or critique?
        if any(re.search(pattern, think_lower) for pattern in self.reflection_words):
            prm_score += 0.1

        # 5. Complexity (0.1): Did it analyze algorithm efficiency?
        if any(re.search(pattern, think_lower) for pattern in self.complexity_words):
            prm_score += 0.1

        return prm_score
'''

# Insert PRM class before format_compliance_reward
if 'class HeuristicPRM' not in content:
    content = content.replace('def format_compliance_reward', prm_class + '\n\ndef format_compliance_reward')

# Update composite_reward to use PRM
target_str = '''    # 5. Length & Efficiency Penalty (Max -0.5 penalty)
    # Discourages rambling in the <think> tags or generating bloated code.
    # We apply a -0.05 penalty for every 1000 characters generated.
    length_penalty = - (len(response) / 1000.0) * 0.05
    length_penalty = max(length_penalty, -0.5) # Cap the penalty at -0.5
    total_reward += length_penalty

    return total_reward'''

replacement_str = '''    # 5. Length & Efficiency Penalty (Max -0.5 penalty)
    # Discourages rambling in the <think> tags or generating bloated code.
    # We apply a -0.05 penalty for every 1000 characters generated.
    length_penalty = - (len(response) / 1000.0) * 0.05
    length_penalty = max(length_penalty, -0.5) # Cap the penalty at -0.5
    total_reward += length_penalty

    # 6. Process Reward Model (PRM) (+0.5 max)
    # Evaluates the internal logic steps independently of the outcome.
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)
        prm = HeuristicPRM()
        prm_score = prm.evaluate(think_content)
        total_reward += prm_score

    return total_reward'''

if '6. Process Reward Model' not in content:
    content = content.replace(target_str, replacement_str)

# Update docstring
doc_target = '    4. Optimization (+0.5 max)'
doc_replacement = '    4. Optimization (+0.5 max)\n    5. Length Penalty (-0.5 max)\n    6. Process Reward (PRM) (+0.5 max)\n    Overall Max Score = 2.5'
if '6. Process Reward' not in content:
    content = content.replace(doc_target, doc_replacement)

with open('src/rewards.py', 'w', encoding='utf-8') as f:
    f.write(content)
