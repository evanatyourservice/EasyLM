#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 --private_git_token <your_private_token> [--force_reinstall]"
    exit 1
}

# Parse command line arguments
FORCE_REINSTALL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --private_git_token) PRIVATE_TOKEN="$2"; shift ;;
        --force_reinstall) FORCE_REINSTALL=true ;;
        *) usage ;;
    esac
    shift
done

# Check if private token is provided
if [ -z "$PRIVATE_TOKEN" ]; then
    usage
fi

# Set variables
TPU_NAME="LLaMA"
ZONE="us-central2-b"
REPO_URL="https://${PRIVATE_TOKEN}@github.com/opooladz/EasyLM.git"

# Function to run command on all TPU VM workers
run_on_all_workers() {
    echo "Running command on all workers: $1"
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "$1"
}

# Clone or update EasyLM repository
run_on_all_workers "
if [ ! -d \"EasyLM\" ]; then
    git clone $REPO_URL
else
    cd EasyLM && git pull
fi
"

# Run setup.sh on all workers
if [ "$FORCE_REINSTALL" = true ]; then
    run_on_all_workers "cd EasyLM && bash scripts/tpu_vm_setup.sh --force_reinstall"
else
    run_on_all_workers "cd EasyLM && bash scripts/tpu_vm_setup.sh"
fi

echo "All operations completed."
