with open('src/sandbox_grader.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == 'stdout = re.sub(r"__RLVR_SCORE__:\d+/\d+:\d+/\d+\\n?", "", stdout)':
        new_lines.append(line)
        new_lines.append('            \n')
        new_lines.append('        coverage_hit = 0\n')
        new_lines.append('        cov_match = re.search(r"__RLVR_COV__:(\d+)", stdout)\n')
        new_lines.append('        if cov_match:\n')
        new_lines.append('            coverage_hit = int(cov_match.group(1))\n')
        new_lines.append('            stdout = re.sub(r"__RLVR_COV__:\d+\\n?", "", stdout)\n')
        new_lines.append('            \n')
        new_lines.append('        import ast\n')
        new_lines.append('        try:\n')
        new_lines.append('            tree = ast.parse(raw_code)\n')
        new_lines.append('            coverage_total = len(set(node.lineno for node in ast.walk(tree) if hasattr(node, "lineno")))\n')
        new_lines.append('        except Exception:\n')
        new_lines.append('            coverage_total = 0\n')
        continue
    new_lines.append(line)

with open('src/sandbox_grader.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
