with open('src/mdp.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '            exec_result = self.grader.grade(\n                extracted_code, problem["test_code"]\n            )'

new = '''            # Step 2.75: Extract Self-Verification Tests
            extracted_sv_tests = self._extract_self_tests(raw_response)

            exec_result = self.grader.grade(
                extracted_code, problem["test_code"], extracted_sv_tests
            )'''

if 'extracted_sv_tests' not in content:
    content = content.replace(target, new)
    with open('src/mdp.py', 'w', encoding='utf-8') as f:
        f.write(content)
