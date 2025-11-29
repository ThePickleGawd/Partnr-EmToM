conda run -n habitat-llm python -m habitat_llm.examples.emtom --config-name game/time_game \
    mode="cli" \
    evaluation.save_video=True \
    game.turn_limit=200 \
    +game.manual_agents=[0,1]
