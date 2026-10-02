with open('src/sandbox_grader.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update ExecutionResult
if 'sv_passed: int = 0' not in content:
    content = content.replace('total_tests: int = 0', 'total_tests: int = 0\n    sv_passed: int = 0\n    sv_total: int = 0')

# 2. Update evaluate_code signature
content = content.replace('def evaluate_code(self, code: str, problem: Dict) -> ExecutionResult:', 'def evaluate_code(self, code: str, problem: Dict, sv_test_code: str = "") -> ExecutionResult:')

# 3. Update grade signature
content = content.replace('def grade(self, code: str, test_code: str) -> ExecutionResult:', 'def grade(self, code: str, test_code: str, sv_test_code: str = "") -> ExecutionResult:')
content = content.replace('full_script = self._build_script(code, test_code)', 'full_script = self._build_script(code, test_code, sv_test_code)')

# 4. Update _build_script
old_build_script = '''    def _build_script(self, code: str, test_code: str) -> str:
        """Combine solution code and test assertions into a single executable, wrapped for partial credit."""
        
        lines = test_code.strip().split('\\n')
        wrapped_lines = []
        wrapped_lines.append('_rlvr_passed = 0')
        wrapped_lines.append('_rlvr_total = 0')
        wrapped_lines.append('import sys')
        
        for line in lines:
            if line.strip():
                wrapped_lines.append('try:')
                wrapped_lines.append(f'    {line}')
                wrapped_lines.append('    _rlvr_passed += 1')
                wrapped_lines.append('except AssertionError:')
                wrapped_lines.append('    pass')
                wrapped_lines.append('except Exception as e:')
                wrapped_lines.append('    pass')
                wrapped_lines.append('_rlvr_total += 1')
                
        wrapped_lines.append('print(f"__RLVR_SCORE__:{_rlvr_passed}/{_rlvr_total}")')
        wrapped_lines.append('if _rlvr_passed < _rlvr_total: sys.exit(1)')
        wrapped_test_code = '\\n'.join(wrapped_lines)
        
        return (
            "# -*- coding: utf-8 -*-\\n"
            "# === AI-GENERATED SOLUTION ===\\n"
            f"{code.strip()}\\n\\n"
            "# === TEST ASSERTIONS ===\\n"
            f"{wrapped_test_code}\\n"
        )'''

new_build_script = '''    def _build_script(self, code: str, test_code: str, sv_test_code: str = "") -> str:
        """Combine solution code and test assertions into a single executable, wrapped for partial credit."""
        
        def wrap_tests(tests_str, passed_var, total_var):
            lines = tests_str.strip().split('\\n')
            wrapped = []
            for line in lines:
                if line.strip():
                    wrapped.append('try:')
                    wrapped.append(f'    {line}')
                    wrapped.append(f'    {passed_var} += 1')
                    wrapped.append('except AssertionError:')
                    wrapped.append('    pass')
                    wrapped.append('except Exception as e:')
                    wrapped.append('    pass')
                    wrapped.append(f'{total_var} += 1')
            return '\\n'.join(wrapped)

        wrapped_lines = []
        wrapped_lines.append('_rlvr_passed = 0')
        wrapped_lines.append('_rlvr_total = 0')
        wrapped_lines.append('_sv_passed = 0')
        wrapped_lines.append('_sv_total = 0')
        wrapped_lines.append('import sys')
        
        if sv_test_code:
            wrapped_lines.append(wrap_tests(sv_test_code, '_sv_passed', '_sv_total'))
            
        if test_code:
            wrapped_lines.append(wrap_tests(test_code, '_rlvr_passed', '_rlvr_total'))
                
        wrapped_lines.append('print(f"__RLVR_SCORE__:{_rlvr_passed}/{_rlvr_total}:{_sv_passed}/{_sv_total}")')
        wrapped_lines.append('if _rlvr_passed < _rlvr_total: sys.exit(1)')
        wrapped_test_code = '\\n'.join(wrapped_lines)
        
        return (
            "# -*- coding: utf-8 -*-\\n"
            "# === AI-GENERATED SOLUTION ===\\n"
            f"{code.strip()}\\n\\n"
            "# === TEST ASSERTIONS ===\\n"
            f"{wrapped_test_code}\\n"
        )'''

content = content.replace(old_build_script, new_build_script)

# 5. Update _execute parsing
target_parse = '''        # Parse the __RLVR_SCORE__ from stdout
        score_match = re.search(r"__RLVR_SCORE__:(\d+)/(\d+)", stdout)
        if score_match:
            tests_passed = int(score_match.group(1))
            total_tests = int(score_match.group(2))
            
            # Remove the score from stdout so it doesn't leak into model feedback
            stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+\\n?", "", stdout)'''

new_parse = '''        # Parse the __RLVR_SCORE__ from stdout
        sv_passed, sv_total = 0, 0
        score_match = re.search(r"__RLVR_SCORE__:(\d+)/(\d+):(\d+)/(\d+)", stdout)
        if score_match:
            tests_passed = int(score_match.group(1))
            total_tests = int(score_match.group(2))
            sv_passed = int(score_match.group(3))
            sv_total = int(score_match.group(4))
            
            # Remove the score from stdout so it doesn't leak into model feedback
            stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+:\d+/\d+\\n?", "", stdout)'''

content = content.replace(target_parse, new_parse)

# 6. Pass sv to ExecutionResult
target_ret = '''            tests_passed=tests_passed,
            total_tests=total_tests,'''
new_ret = '''            tests_passed=tests_passed,
            total_tests=total_tests,
            sv_passed=sv_passed,
            sv_total=sv_total,'''
content = content.replace(target_ret, new_ret)

with open('src/sandbox_grader.py', 'w', encoding='utf-8') as f:
    f.write(content)
