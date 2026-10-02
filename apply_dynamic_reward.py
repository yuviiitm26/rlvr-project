with open('src/rewards.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = '''def composite_reward(
    exec_result: "ExecutionResult",
    turn: int,
    response: str,
    config: RewardConfig,
) -> float:'''

new1 = '''def composite_reward(
    exec_result: "ExecutionResult",
    turn: int,
    response: str,
    config: RewardConfig,
    current_discount: float = None,
) -> float:'''

target2 = '''        # Discount logic score by turn to penalize multi-turn thrashing
        discounted_logic = logic_score * (config.discount_gamma ** (turn - 1))'''

new2 = '''        # Discount logic score by turn to penalize multi-turn thrashing
        if current_discount is None:
            # Fallback to static discount
            discounted_logic = logic_score * (config.discount_gamma ** (turn - 1))
        else:
            # Use dynamic adaptive discount
            discounted_logic = logic_score * current_discount'''

content = content.replace(target1, new1)
content = content.replace(target2, new2)

with open('src/rewards.py', 'w', encoding='utf-8') as f:
    f.write(content)
