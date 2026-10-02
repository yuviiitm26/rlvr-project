with open('src/mdp.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''            messages.append({"role": "assistant", "content": raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": self._build_retry_prompt(
                        exec_result, turn_num
                    ),
                }
            )'''

new = '''            # --- Adaptive Error Discounting (Dynamic MDP) ---
            # Update the discount factor for the NEXT turn based on the severity of the error
            if exec_result.error_type == "Timeout":
                current_discount *= 0.5   # Harsh penalty for infinite loops
            elif exec_result.error_type in ["Syntax_Error", "Indentation_Error", "Name_Error"]:
                current_discount *= 0.8   # Moderate penalty for basic syntax errors
            elif exec_result.error_type == "Assertion_Failure":
                current_discount *= 0.9   # Gentle penalty for logic errors
            else:
                current_discount *= 0.95  # Very gentle penalty for other runtime errors

            messages.append({"role": "assistant", "content": raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": self._build_retry_prompt(
                        exec_result, turn_num
                    ),
                }
            )'''

if 'current_discount *= 0.5' not in content:
    content = content.replace(target, new)
    with open('src/mdp.py', 'w', encoding='utf-8') as f:
        f.write(content)
