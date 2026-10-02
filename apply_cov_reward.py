with open('src/rewards.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''    if total_tests > 0:
        logic_score = (passed_tests / total_tests) * 1.0
        # Discount logic score by turn to penalize multi-turn thrashing'''

new = '''    if total_tests > 0:
        logic_score = (passed_tests / total_tests) * 1.0
        
        # --- Coverage Reward Scaling ---
        cov_hit = getattr(exec_result, "coverage_hit", 0)
        cov_total = getattr(exec_result, "coverage_total", 0)
        
        # If the code passed tests but has dead code, penalize logic score
        if cov_total > 0 and passed_tests == total_tests:
            cov_ratio = min(cov_hit / cov_total, 1.0)
            logic_score = logic_score * cov_ratio
            
        # Discount logic score by turn to penalize multi-turn thrashing'''

if 'cov_hit =' not in content:
    content = content.replace(target, new)
    with open('src/rewards.py', 'w', encoding='utf-8') as f:
        f.write(content)
