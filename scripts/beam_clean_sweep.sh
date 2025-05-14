#!/bin/bash

# List of available CUDA devices.
# Adjust these device IDs according to your hardware.
CUDA_DEVICES=(1 2 3 4 5 6 7)

# Define an array of parameters for the different runs.
# For example, these could be different noise levels or any custom string.
# PARAMS=("180" "190" "200" "210" "220" "230" "240" "250") # ("100" "110" "120" "130" "140" "150" "160" "170") # ("220" "230" "240" "250")
# PARAMS=("0.399" "0.266" "0.2" "0.16" "0.133" "0.114" "0.1")
PARAMS=("0.296" "0.222" "0.178" "0.148" "0.127" "0.111" "0.099") # Laplace
# Activate your conda environment.
source activate /home/kaan/.conda/envs/blind_inversion

# Iterate over each parameter in the PARAMS list.
# If there are more parameters than devices, the modulo operator (%) cycles through CUDA_DEVICES.
for idx in "${!PARAMS[@]}"; do
    # Calculate the appropriate CUDA device index (cycling through if necessary)
    device=${CUDA_DEVICES[$(( idx % ${#CUDA_DEVICES[@]} ))]}
    # Get the current parameter value.
    parameter=${PARAMS[$idx]}
    
    # Set the CUDA device for this job.
    export CUDA_VISIBLE_DEVICES=$device
    echo "Launching job $((idx+1)) on CUDA device $device with parameter: $parameter"
    
    # Run the python training command in the background.
    # python beam_clean_decode.py privacy.const_noise_params.gauss.std=$parameter &
    python beam_clean_decode.py privacy.const_noise_params.laplace.scale=$parameter attack.estimated_param_val=$parameter &
    # Wait for a short time
    sleep 5
done

# Wait for all background jobs to finish.
wait
echo "All jobs have completed."
