python -m habitat_llm.examples.emtom --config-name baselines/decentralized_emtom.yaml \
    mode="cli" \
    evaluation.save_video=True \
    instruction="find the pan and agree on who should pick it up" \
    llm@evaluation.agents.agent_0.planner.plan_config.llm=openai_chat \
    llm@evaluation.agents.agent_1.planner.plan_config.llm=openai_chat \
    evaluation.agents.agent_0.planner.plan_config.llm.use_image_input=True \
    evaluation.agents.agent_1.planner.plan_config.llm.use_image_input=True \
    evaluation.agents.agent_0.planner.plan_config.llm.save_prompt_images=True \
    evaluation.agents.agent_1.planner.plan_config.llm.save_prompt_images=True \
    trajectory.save=True \
    'trajectory.save_options=["rgb"]' \
    habitat.environment.max_episode_steps=10000 \
