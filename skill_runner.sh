HYDRA_FULL_ERROR=1 python3 -m habitat_llm.examples.skill_runner \
  +skill_runner_show_videos=false \
  habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
  +skill_runner_episode_id="334" \
  agent@evaluation.agents.agent_0.config=oracle_rearrange_agent \
  # +skill_runner_show_topdown=true \
