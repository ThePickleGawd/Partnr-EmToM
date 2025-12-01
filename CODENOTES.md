# CODENOTES

Quick reminders for running planners with OpenAI:

1) Make sure the OpenAI key is available in the shell. In this repo we keep it in `.env`, so source it before running:
   ```bash
   set -a && source .env && set +a
   ```
   (Keep `.env` out of git; `.gitignore` already lists it.)

2) Activate the conda env and run a script, e.g. bomb game:
   ```bash
   source /home/dylanlu/miniconda3/etc/profile.d/conda.sh
   conda activate habitat-llm
   set -a && source .env && set +a
   ./bomb_game_run.sh
   ```

3) If you need a timeout while testing:
   ```bash
   source /home/dylanlu/miniconda3/etc/profile.d/conda.sh
   conda activate habitat-llm
   set -a && source .env && set +a
   timeout 30s ./bomb_game_run.sh
   ```

4) If the key is in `.bashrc`, remember that non-interactive shells skip most of it due to the interactive guard. Use the `.env` approach above to ensure the key is present for Hydra runs.

5) Inspect the latest run/log quickly:
   ```bash
   latest=$(ls -t outputs/habitat_llm | head -n1)
   tail -n 40 outputs/habitat_llm/$latest/habitat_llm.log
   cat outputs/habitat_llm/$latest/.hydra/overrides.yaml
   ```
