with open('src/rewards.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''    return total_reward'''

new_ast_logic = '''    # 8. AST Complexity (Pythonic Code) Bonus/Penalty (-0.3 to +0.2)
    # Rewards flat, readable code and penalizes deeply nested spaghetti code.
    if exec_result.passed and passed_tests == total_tests and total_tests > 0:
        import ast
        try:
            tree = ast.parse(exec_result.raw_code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler, ast.With, ast.ListComp, ast.DictComp)):
                    complexity += 1
            
            if complexity <= 3:
                total_reward += 0.2 # Very flat, elegant
            elif complexity <= 5:
                total_reward += 0.1 # Standard, clean
            elif complexity > 10:
                total_reward -= 0.3 # High cyclomatic complexity (Spaghetti)
            elif complexity > 7:
                total_reward -= 0.1 # Starting to get messy
        except Exception:
            pass # Ignore parsing errors here (handled by syntax reward)

    return total_reward'''

# Replace only the LAST occurrence of `    return total_reward` which is in composite_reward
if '8. AST Complexity' not in content:
    # Reverse replace to only get the last one
    content = content[::-1].replace(target[::-1], new_ast_logic[::-1], 1)[::-1]
    
doc_target = '''    7. Self-Verification (+0.5 max)
    Overall Max Score = 3.0'''
doc_new = '''    7. Self-Verification (+0.5 max)
    8. AST Complexity (-0.3 to +0.2)
    Overall Max Score = 3.2'''
    
if '8. AST Complexity' not in content:
    pass # Wait, if it failed the first condition it skips.
    
content = content.replace(doc_target, doc_new)

with open('src/rewards.py', 'w', encoding='utf-8') as f:
    f.write(content)
