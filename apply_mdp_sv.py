with open('src/mdp.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update SYSTEM_PROMPT
old_prompt = '''    SYSTEM_PROMPT = (
        "You are a Python coding assistant. Your task is to write correct "
        "Python code that passes all test assertions.\\n\\n"
        "RULES:\\n"
        "1. Wrap your code in a ```python code block.\\n"
        "2. Before writing code, reason about the problem inside "
        "<think></think> tags.\\n"
        "3. Write ONLY the solution function(s). Do NOT include test code.\\n"
        "4. If you receive error feedback, analyze it carefully and fix "
        "your solution.\\n"
    )'''

new_prompt = '''    SYSTEM_PROMPT = (
        "You are a Python coding assistant. Your task is to write correct "
        "Python code that passes all test assertions.\\n\\n"
        "RULES:\\n"
        "1. Before writing code, reason about the problem inside <think></think> tags.\\n"
        "2. INSIDE your <think> tags, you MUST write your own test cases to verify your logic. "
        "Write these tests in a ```python block containing `assert` statements.\\n"
        "3. AFTER the <think> tags, write ONLY the final solution function(s) wrapped in a ```python block. "
        "Do NOT include test code in the final solution block.\\n"
        "4. If you receive error feedback, analyze it carefully and fix your solution.\\n"
    )'''

content = content.replace(old_prompt, new_prompt)

# 2. Add _extract_self_tests before _extract_code
extract_sv = '''    def _extract_self_tests(self, response: str) -> str:
        """Extracts self-verification assertions written inside the <think> block."""
        import re
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        if not think_match:
            return ""
        
        think_content = think_match.group(1)
        blocks = re.findall(r'```python(.*?)```', think_content, re.DOTALL | re.IGNORECASE)
        
        sv_code = []
        for b in blocks:
            if 'assert ' in b:
                sv_code.append(b.strip())
        return "\\n".join(sv_code)

    def _extract_code('''
content = content.replace('    def _extract_code(', extract_sv)

# 3. Update Grade call in run_episode
old_grade = '''            # Step 3: Grade the code
            exec_result = self.grader.grade(
                extracted_code, problem["test_code"]
            )'''

new_grade = '''            # Step 2.75: Extract Self-Verification Tests
            extracted_sv_tests = self._extract_self_tests(raw_response)

            # Step 3: Grade the code
            exec_result = self.grader.grade(
                extracted_code, problem["test_code"], extracted_sv_tests
            )'''

content = content.replace(old_grade, new_grade)

with open('src/mdp.py', 'w', encoding='utf-8') as f:
    f.write(content)
