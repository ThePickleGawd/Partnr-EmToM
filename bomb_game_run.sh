python -m habitat_llm.examples.emtom --config-name game/bomb_game \
    mode="cli" \
    evaluation.save_video=True \
    +game.manual_agents=[] \
    +game.manual_obs_popup=true \
    # habitat.environment.max_episode_steps=10000 \
