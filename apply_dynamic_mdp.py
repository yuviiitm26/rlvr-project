with open('src/mdp.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = '''        episode_start = time.monotonic()
        seen_codes = set()

        for turn_num in range(1, self.max_turns + 1):'''

new1 = '''        episode_start = time.monotonic()
        seen_codes = set()
        current_discount = 1.0

        for turn_num in range(1, self.max_turns + 1):'''

target2 = '''            # Step 4: Compute reward
            reward = self._compute_reward(
                exec_result, turn_num, raw_response
            )'''

new2 = '''            # Step 4: Compute reward
            reward = self._compute_reward(
                exec_result, turn_num, raw_response, current_discount
            )'''

target3 = '''    def _compute_reward(
        self, exec_result: "ExecutionResult", turn: int, raw_response: str
    ) -> float:
        """
        Compute the reward for this turn using the configured strategy.

        Delegates to the rewards module based on RewardConfig settings.
        """
        return composite_reward(
            exec_result=exec_result,
            turn=turn,
            response=raw_response,
            config=self.reward_config,
        )'''

new3 = '''    def _compute_reward(
        self, exec_result: "ExecutionResult", turn: int, raw_response: str, current_discount: float = None
    ) -> float:
        """
        Compute the reward for this turn using the configured strategy.

        Delegates to the rewards module based on RewardConfig settings.
        """
        return composite_reward(
            exec_result=exec_result,
            turn=turn,
            response=raw_response,
            config=self.reward_config,
            current_discount=current_discount,
        )'''

target4 = '''            # Step 7: Append feedback for next turn
            # This is the MDP state transition: the model's failed attempt
            # and the grader's feedback become part of the new state.
            messages.append({"role": "assistant", "content": raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": self._build_retry_prompt(
                        exec_result, turn_num
                    ),
                }
            )'''

new4 = '''            # Step 7: Append feedback for next turn
            # This is the MDP state transition: the model's failed attempt
            # and the grader's feedback become part of the new state.
            
            # --- Adaptive Error Discounting (Dynamic MDP) ---
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

content = content.replace(target1, new1)
content = content.replace(target2, new2)
content = content.replace(target3, new3)
content = content.replace(target4, new4)

with open('src/mdp.py', 'w', encoding='utf-8') as f:
    f.write(content)
