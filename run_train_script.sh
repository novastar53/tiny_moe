make uv
source $HOME/.local/bin/env
make cuda
hf auth login --token $HF_TOKEN --no-add-to-git-credential
nohup uv run train.py > output.log 2>&1 &
cd training_runs/Tiny_MoE
hf upload vikramp/Tiny_Moe_2 .