#!/bin/bash
# remote_train.sh - Remote training script with HuggingFace integration
#
# Usage: ./remote_train.sh [-b BRANCH] <ssh-host> [command]
#
# Options:
#   -b BRANCH  - Git branch to checkout before training (default: main)
#
# Commands:
#   start    - Clone repo, install deps, start training, stream logs (default)
#   attach   - Attach to existing tmux session
#   stream   - Stream logs from running training
#   status   - Check if training is running
#   stop     - Stop the training gracefully and upload to HuggingFace
#   upload   - Upload training outputs to HuggingFace
#
# Environment Variables:
#   HF_TOKEN - HuggingFace API token (required for start, stop, upload)

set -e

# Configuration
REMOTE_DIR="~/tiny_moe"
REPO_URL="https://github.com/novastar53/tiny_moe.git"
TMUX_SESSION="train"
HF_REPO="vikramp/Tiny_Moe_2"
LOG_FILE="training.log"
BRANCH="main"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse options
while getopts "b:" opt; do
    case $opt in
        b)
            BRANCH="$OPTARG"
            ;;
        \?)
            echo -e "${RED}Invalid option: -$OPTARG${NC}" >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# Parse positional arguments
SSH_HOST=${1:-}
COMMAND=${2:-start}

if [ -z "$SSH_HOST" ]; then
    echo -e "${RED}Error: SSH host is required${NC}"
    echo "Usage: $0 [-b BRANCH] <ssh-host> [command]"
    echo "Commands: start, attach, stream, status, stop, upload"
    exit 1
fi

# Helper function to run commands on remote
remote_exec() {
    ssh "$SSH_HOST" "$@"
}

# Helper function to run commands in remote tmux session
remote_tmux_exec() {
    remote_exec "tmux send-keys -t $TMUX_SESSION '$1' Enter"
}

cmd_start() {
    echo -e "${GREEN}Starting training on $SSH_HOST (branch: $BRANCH)...${NC}"

    # Check for HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}Warning: HF_TOKEN not set. HuggingFace upload will not work.${NC}"
    fi

    # Setup and start training
    remote_exec "bash -l" << REMOTE_SCRIPT
set -e

# Expand REMOTE_DIR
REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)

# Clone or update repo
if [ -d "\$REMOTE_DIR_EXPANDED" ]; then
    echo "Updating existing repo..."
    cd "\$REMOTE_DIR_EXPANDED"
    git fetch origin
else
    echo "Cloning repo..."
    git clone $REPO_URL "\$REMOTE_DIR_EXPANDED"
    cd "\$REMOTE_DIR_EXPANDED"
fi

# Checkout and update the specified branch
echo "Switching to branch '$BRANCH'..."
git checkout $BRANCH
git reset --hard origin/$BRANCH

# Source uv environment (if already installed)
if [ -f "\$HOME/.local/bin/env" ]; then
    source "\$HOME/.local/bin/env"
fi

# Install tmux if not available
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# Install dependencies
echo "Installing dependencies..."
make uv || true

# Source uv environment again (in case make uv just installed it)
if [ -f "\$HOME/.local/bin/env" ]; then
    source "\$HOME/.local/bin/env"
fi

make hf || true
make cuda

# Authenticate with HuggingFace if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Authenticating with HuggingFace..."
    hf auth login --token "$HF_TOKEN" --no-add-to-git-credential
fi

# Kill existing tmux session if it exists
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

# Create new tmux session and start training
echo "Starting training in tmux session '$TMUX_SESSION'..."
tmux new-session -d -s $TMUX_SESSION -c "\$REMOTE_DIR_EXPANDED"
tmux send-keys -t $TMUX_SESSION "source \$HOME/.local/bin/env 2>/dev/null; uv run python train.py 2>&1 | tee $LOG_FILE" Enter

echo "Training started successfully!"
REMOTE_SCRIPT

    echo -e "${GREEN}Training started. Streaming logs...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to disconnect (training will continue in background)${NC}"
    echo ""

    # Stream logs
    cmd_stream
}

cmd_attach() {
    echo -e "${GREEN}Attaching to tmux session on $SSH_HOST...${NC}"
    ssh -t "$SSH_HOST" "tmux attach-session -t $TMUX_SESSION"
}

cmd_stream() {
    echo -e "${GREEN}Streaming logs from $SSH_HOST...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to disconnect (training will continue)${NC}"
    echo ""

    # Use tail -f to stream logs, with trap to handle disconnect gracefully
    ssh "$SSH_HOST" "cd \$(eval echo $REMOTE_DIR) && tail -f $LOG_FILE 2>/dev/null" || true
}

cmd_status() {
    echo -e "${GREEN}Checking training status on $SSH_HOST...${NC}"

    remote_exec "bash -l" << REMOTE_SCRIPT
REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)

# Check if tmux session exists
if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
    echo "Tmux session '$TMUX_SESSION': RUNNING"
else
    echo "Tmux session '$TMUX_SESSION': NOT RUNNING"
fi

# Check if training process is running
if pgrep -f "train.py" > /dev/null; then
    echo "Training process: RUNNING"
    echo ""
    echo "Process info:"
    ps aux | grep "[t]rain.py" | head -5
else
    echo "Training process: NOT RUNNING"
fi

# Show last few lines of log if it exists
if [ -f "\$REMOTE_DIR_EXPANDED/$LOG_FILE" ]; then
    echo ""
    echo "Last 10 lines of log:"
    tail -10 "\$REMOTE_DIR_EXPANDED/$LOG_FILE"
fi
REMOTE_SCRIPT
}

cmd_stop() {
    echo -e "${GREEN}Stopping training on $SSH_HOST...${NC}"

    # Check for HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}Warning: HF_TOKEN not set. Skipping HuggingFace upload.${NC}"
    fi

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e
REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)

# Check if tmux session exists
if ! tmux has-session -t $TMUX_SESSION 2>/dev/null; then
    echo "No training session found."
    exit 0
fi

# Send Ctrl+C to gracefully stop training
echo "Sending interrupt signal to training..."
tmux send-keys -t $TMUX_SESSION C-c

# Wait for process to exit (with timeout)
echo "Waiting for training to save checkpoint and exit..."
for i in {1..60}; do
    if ! pgrep -f "train.py" > /dev/null; then
        echo "Training stopped."
        break
    fi
    sleep 2
    echo "  Still waiting... (\$i)"
done

# Force kill if still running
if pgrep -f "train.py" > /dev/null; then
    echo "Force killing training process..."
    pkill -9 -f "train.py" || true
fi

# Kill tmux session
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
echo "Tmux session closed."
REMOTE_SCRIPT

    # Upload to HuggingFace if token is set
    if [ -n "$HF_TOKEN" ]; then
        echo ""
        echo -e "${GREEN}Uploading to HuggingFace...${NC}"
        cmd_upload
    fi

    echo -e "${GREEN}Training stopped successfully.${NC}"
}

cmd_upload() {
    echo -e "${GREEN}Uploading training outputs to HuggingFace...${NC}"

    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: HF_TOKEN is required for upload${NC}"
        exit 1
    fi

    remote_exec "bash -l" << REMOTE_SCRIPT
set -e
REMOTE_DIR_EXPANDED=\$(eval echo $REMOTE_DIR)

# Source uv environment
if [ -f "\$HOME/.local/bin/env" ]; then
    source "\$HOME/.local/bin/env"
fi

# Authenticate with HuggingFace
hf auth login --token "$HF_TOKEN" --no-add-to-git-credential

# Navigate to training outputs
cd "\$REMOTE_DIR_EXPANDED/training_runs/Tiny_MoE"

# Upload to HuggingFace
echo "Uploading to $HF_REPO..."
hf upload $HF_REPO .

echo "Upload complete!"
REMOTE_SCRIPT

    echo -e "${GREEN}Upload to HuggingFace complete.${NC}"
}

# Execute the requested command
case "$COMMAND" in
    start)
        cmd_start
        ;;
    attach)
        cmd_attach
        ;;
    stream)
        cmd_stream
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    upload)
        cmd_upload
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo "Available commands: start, attach, stream, status, stop, upload"
        exit 1
        ;;
esac
