#!/bin/bash

# Function to run a command and return its output
run_command() {
    output=$(eval "$1" 2>&1)
    echo "$output"
}

# Run lsof command and store the output
lsof_output=$(run_command "gcloud compute tpus tpu-vm ssh LLaMA --zone us-central2-b --worker=all --command 'sudo lsof -w /dev/accel0'")

# Extract PIDs
pids=$(echo "$lsof_output" | grep llama_tra | awk '{print $2}' | sort -u)

if [ -z "$pids" ]; then
    echo "No processes found using /dev/accel0"
else
    # Kill each process
    for pid in $pids; do
        echo "Attempting to kill process $pid..."
        kill_output=$(run_command "gcloud compute tpus tpu-vm ssh LLaMA --zone us-central2-b --worker=all --command 'sudo kill $pid'")
        echo "$kill_output"
    done

    # Verify if processes were killed
    verify_output=$(run_command "gcloud compute tpus tpu-vm ssh LLaMA --zone us-central2-b --worker=all --command 'sudo lsof -w /dev/accel0'")
    
    if echo "$verify_output" | grep -q "llama_tra"; then
        echo "Some processes may still be running. You might need to use 'kill -9' or restart the TPU VM."
    else
        echo "All processes successfully terminated."
    fi
fi
