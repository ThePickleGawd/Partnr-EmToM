HYDRA_FULL_ERROR=1 python3 -m habitat_llm.examples.skill_runner \
  hydra.run.dir="." \
  +skill_runner_show_topdown=false \
  +skill_runner_make_video=true \
  habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/val_mini.json.gz \
  +skill_runner_episode_id="334"
