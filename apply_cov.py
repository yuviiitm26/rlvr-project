with open('src/sandbox_grader.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update ExecutionResult
if 'coverage_hit: int = 0' not in content:
    content = content.replace('sv_total: int = 0', 'sv_total: int = 0\n    coverage_hit: int = 0\n    coverage_total: int = 0')

# 2. Update _build_script
old_build_script = '''    def _build_script(self, code: str, test_code: str, sv_test_code: str = "") -> str:
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

        header_lines = [
            "# -*- coding: utf-8 -*-",
            "import sys",
            "_rlvr_exec = set()",
            "def _rlvr_trace(frame, event, arg):",
            "    if event == 'line' and frame.f_code.co_filename == __file__:",
            "        _rlvr_exec.add(frame.f_lineno)",
            "    return _rlvr_trace",
            "sys.settrace(_rlvr_trace)",
            "# === AI-GENERATED SOLUTION ==="
        ]
        start_line = len(header_lines) + 1
        ai_lines = code.strip().split('\\n')
        end_line = start_line + len(ai_lines) - 1

        wrapped_lines = []
        wrapped_lines.append('_rlvr_passed = 0')
        wrapped_lines.append('_rlvr_total = 0')
        wrapped_lines.append('_sv_passed = 0')
        wrapped_lines.append('_sv_total = 0')
        
        if sv_test_code:
            wrapped_lines.append(wrap_tests(sv_test_code, '_sv_passed', '_sv_total'))
            
        if test_code:
            wrapped_lines.append(wrap_tests(test_code, '_rlvr_passed', '_rlvr_total'))
                
        wrapped_lines.append(f'__cov_hit = len([l for l in _rlvr_exec if {start_line} <= l <= {end_line}])')
        wrapped_lines.append('print(f"__RLVR_SCORE__:{_rlvr_passed}/{_rlvr_total}:{_sv_passed}/{_sv_total}")')
        wrapped_lines.append('print(f"__RLVR_COV__:{__cov_hit}")')
        wrapped_lines.append('if _rlvr_passed < _rlvr_total: sys.exit(1)')
        wrapped_test_code = '\\n'.join(wrapped_lines)
        
        return (
            "\\n".join(header_lines) + "\\n" +
            f"{code.strip()}\\n\\n" +
            "# === TEST ASSERTIONS ===\\n" +
            f"{wrapped_test_code}\\n"
        )'''

if 'sys.settrace(_rlvr_trace)' not in content:
    content = content.replace(old_build_script, new_build_script)

# 3. Update _execute
target_exec = '''        # Parse the __RLVR_SCORE__ from stdout
        sv_passed, sv_total = 0, 0
        score_match = re.search(r"__RLVR_SCORE__:(\d+)/(\d+):(\d+)/(\d+)", stdout)
        if score_match:
            tests_passed = int(score_match.group(1))
            total_tests = int(score_match.group(2))
            sv_passed = int(score_match.group(3))
            sv_total = int(score_match.group(4))
            
            # Remove the score from stdout so it doesn't leak into model feedback
            stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+:\d+/\d+\\n?", "", stdout)'''

new_exec = '''        # Parse the __RLVR_SCORE__ and __RLVR_COV__ from stdout
        sv_passed, sv_total = 0, 0
        score_match = re.search(r"__RLVR_SCORE__:(\d+)/(\d+):(\d+)/(\d+)", stdout)
        if score_match:
            tests_passed = int(score_match.group(1))
            total_tests = int(score_match.group(2))
            sv_passed = int(score_match.group(3))
            sv_total = int(score_match.group(4))
            stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+:\d+/\d+\\n?", "", stdout)
            
        coverage_hit = 0
        cov_match = re.search(r"__RLVR_COV__:(\d+)", stdout)
        if cov_match:
            coverage_hit = int(cov_match.group(1))
            stdout = re.sub(r"__RLVR_COV__:\d+\\n?", "", stdout)
            
        import ast
        try:
            tree = ast.parse(raw_code)
            coverage_total = len(set(node.lineno for node in ast.walk(tree) if hasattr(node, "lineno")))
        except Exception:
            coverage_total = 0'''

if '__RLVR_COV__' not in content:
    content = content.replace(target_exec, new_exec)

target_ret = '''            tests_passed=tests_passed,
            total_tests=total_tests,
            sv_passed=sv_passed,
            sv_total=sv_total,'''
new_ret = '''            tests_passed=tests_passed,
            total_tests=total_tests,
            sv_passed=sv_passed,
            sv_total=sv_total,
            coverage_hit=coverage_hit,
            coverage_total=coverage_total,'''
if 'coverage_hit=coverage_hit' not in content:
    content = content.replace(target_ret, new_ret)

with open('src/sandbox_grader.py', 'w', encoding='utf-8') as f:
    f.write(content)
