with open('src/rewards.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''    # 6. Process Reward Model (PRM) (+0.5 max)
    # Evaluates the internal logic steps independently of the outcome.
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)
        prm = HeuristicPRM()
        prm_score = prm.evaluate(think_content)
        total_reward += prm_score

    return total_reward'''

new = '''    # 6. Process Reward Model (PRM) (+0.5 max)
    # Evaluates the internal logic steps independently of the outcome.
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)
        prm = HeuristicPRM()
        prm_score = prm.evaluate(think_content)
        total_reward += prm_score

    # 7. Self-Verification Reward (+0.5 max)
    # Rewards the model if it successfully wrote its own test cases AND passed the real tests.
    sv_passed = getattr(exec_result, "sv_passed", 0)
    sv_total = getattr(exec_result, "sv_total", 0)
    
    if exec_result.passed and passed_tests == total_tests and total_tests > 0:
        if sv_total > 0:
            # Massive bonus for self-verification
            if sv_passed == sv_total and sv_total >= 2:
                total_reward += 0.5
            else:
                # Partial bonus
                total_reward += (sv_passed / sv_total) * 0.2

    return total_reward'''

doc_target = '''    4. Optimization (+0.5 max)
    5. Length Penalty (-0.5 max)
    6. Process Reward (PRM) (+0.5 max)
    Overall Max Score = 2.5'''

doc_new = '''    4. Optimization (+0.5 max)
    5. Length Penalty (-0.5 max)
    6. Process Reward (PRM) (+0.5 max)
    7. Self-Verification (+0.5 max)
    Overall Max Score = 3.0'''

if '7. Self-Verification Reward' not in content:
    content = content.replace(target, new)
    content = content.replace(doc_target, doc_new)
    with open('src/rewards.py', 'w', encoding='utf-8') as f:
        f.write(content)
