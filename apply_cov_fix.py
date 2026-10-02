with open('src/sandbox_grader.py', 'r', encoding='utf-8') as f:
    content = f.read()

target_exec = '''        # Parse the __RLVR_SCORE__ from stdout
        sv_passed, sv_total = 0, 0
        score_match = re.search(r"__RLVR_SCORE__:(\d+)/(\d+):(\d+)/(\d+)", stdout)
        if score_match:
            tests_passed = int(score_match.group(1))
            total_tests = int(score_match.group(2))
            sv_passed = int(score_match.group(3))
            sv_total = int(score_match.group(4))
            
            # Remove the score from stdout so it doesn't leak into model feedback
            stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+:\d+/\d+\n?", "", stdout)'''

new_exec = '''        # Parse the __RLVR_SCORE__ and __RLVR_COV__ from stdout
        sv_passed, sv_total = 0, 0
        score_match = re.search(r"__RLVR_SCORE__:(\d+)/(\d+):(\d+)/(\d+)", stdout)
        if score_match:
            tests_passed = int(score_match.group(1))
            total_tests = int(score_match.group(2))
            sv_passed = int(score_match.group(3))
            sv_total = int(score_match.group(4))
            stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+:\d+/\d+\n?", "", stdout)
            
        coverage_hit = 0
        cov_match = re.search(r"__RLVR_COV__:(\d+)", stdout)
        if cov_match:
            coverage_hit = int(cov_match.group(1))
            stdout = re.sub(r"__RLVR_COV__:\d+\n?", "", stdout)
            
        import ast
        try:
            tree = ast.parse(raw_code)
            coverage_total = len(set(node.lineno for node in ast.walk(tree) if hasattr(node, "lineno")))
        except Exception:
            coverage_total = 0'''

if '__RLVR_COV__' not in content:
    content = content.replace(target_exec, new_exec)
    with open('src/sandbox_grader.py', 'w', encoding='utf-8') as f:
        f.write(content)
