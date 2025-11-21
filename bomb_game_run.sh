python -m habitat_llm.examples.emtom --config-name game/bomb_game \
    mode="cli" \
    evaluation.save_video=True \
    habitat.environment.max_episode_steps=10000
