#!/bin/bash

# Parse command line arguments
FORCE_REINSTALL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --force_reinstall) FORCE_REINSTALL=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Update system packages
sudo apt-get update && sudo apt-get install -y build-essential python-is-python3 tmux htop git nodejs bmon p7zip-full nfs-common

# Set up Python environment
if ! command -v python3.10 &> /dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install -y python3.10 python3.10-venv
fi
if [ ! -d "$HOME/venv" ] || [ "$FORCE_REINSTALL" = true ]; then
    python3.10 -m venv $HOME/venv
fi
source $HOME/venv/bin/activate

# Create tpu_requirements.txt if it doesn't exist
REQUIREMENTS_FILE="$HOME/tpu_requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    cat > "$REQUIREMENTS_FILE" <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.4.28
flax==0.8.3
optax==0.2.2
einops
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.0
transformers==4.41.0
datasets==2.19.1
tqdm
requests
typing-extensions
mlxu>=0.1.13
sentencepiece
pydantic
fastapi
uvicorn
gradio
EndOfFile
    echo "Created $REQUIREMENTS_FILE"
fi

# Handle Python dependencies
HASH_FILE="$HOME/.requirements_hash"
CURRENT_HASH=$(md5sum "$REQUIREMENTS_FILE" | awk '{ print $1 }')
LAST_UPDATE_FILE="$HOME/.last_pip_update"

if [ "$FORCE_REINSTALL" = true ] || [ ! -f "$HASH_FILE" ] || [ "$CURRENT_HASH" != "$(cat "$HASH_FILE")" ] || [ ! -f "$LAST_UPDATE_FILE" ] || [ $(($(date +%s) - $(stat -c %Y "$LAST_UPDATE_FILE"))) -gt 86400 ]; then
    pip install --upgrade -r "$REQUIREMENTS_FILE"
    echo "$CURRENT_HASH" > "$HASH_FILE"
    touch "$LAST_UPDATE_FILE"
else
    echo "Python dependencies are up to date. Skipping installation."
fi

# Set up .vimrc
cat > $HOME/.vimrc <<- EndOfFile
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set backspace=indent,eol,start
syntax on
EndOfFile

# Set up .tmux.conf
cat > $HOME/.tmux.conf <<- EndOfFile
bind r source-file ~/.tmux.conf
set -g prefix C-a
set -g set-titles on
set -g set-titles-string '#(whoami)::#h::#(curl ipecho.net/plain;echo)'
set -g default-terminal "screen-256color"
set -g status-bg white
set -g status-fg black
set -g status-interval 5
set -g status-left-length 90
set -g status-right-length 60
set -g status-justify left
unbind-key C-o
bind -n C-o prev
unbind-key C-p
bind -n C-p next
unbind-key C-w
bind -n C-w new-window
unbind-key C-j
bind -n C-j select-pane -D
unbind-key C-k
bind -n C-k select-pane -U
unbind-key C-h
bind -n C-h select-pane -L
unbind-key C-l
bind -n C-l select-pane -R
unbind-key C-e
bind -n C-e split-window -h
unbind-key C-q
bind -n C-q split-window -v
unbind '"'
unbind %
unbind-key u
bind-key u split-window -h
unbind-key i
bind-key i split-window -v
EndOfFile

# Set up htop configuration
mkdir -p $HOME/.config/htop
cat > $HOME/.config/htop/htoprc <<- EndOfFile
fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=1
hide_threads=0
hide_kernel_threads=1
hide_userland_threads=1
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
tree_view=0
header_margin=1
detailed_cpu_time=0
cpu_count_from_zero=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
delay=15
left_meters=CPU Memory Swap
left_meter_modes=1 1 1
right_meters=Tasks LoadAverage Uptime
right_meter_modes=2 2 2
EndOfFile

echo "Setup completed."
