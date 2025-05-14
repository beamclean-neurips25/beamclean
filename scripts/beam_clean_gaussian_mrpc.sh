#!/bin/bash
CUDA_DEVICES=(0 1 2 3 4 5 6 7) 

echo "Launching job on CUDA device $device"
PARAMS=("0.423" "0.282" "0.211" "0.169" "0.141" "0.121" "0.106")
# Run the python training command in the background.
for idx in "${!PARAMS[@]}"; do
    # Calculate the appropriate CUDA device index (cycling through if necessary)
    device=${CUDA_DEVICES[$(( idx % ${#CUDA_DEVICES[@]} ))]}

    export CUDA_VISIBLE_DEVICES=$device
    echo "Launching job $((idx+1)) on CUDA device $device"
    param=${PARAMS[$idx]}

    python beam_clean.py data=mrpc data.truncated_seq_len=32 \
                                data.batch_size=2 \
                                run.beam_clean=true \
                                run.nearest_neighbor=true \
                                data.subset_size=50 \
                                privacy.embedding_model_path="../LLama-3.2-1B-Instruct" \
                                privacy.const_noise_params.noise_dist=gaussian \
                                privacy.const_noise_params.gaussian.std=$param \
                                beam_clean.candidate_token_ids_path=null \
                                beam_clean.prior_model="../LLama-3.2-1B-Instruct" \
                                beam_clean.surrogate_model=gaussian \
                                beam_clean.initial_param_val=$param \
                                beam_clean.num_epochs=50 \
                                beam_clean.beam_width=20 \
                                beam_clean.vocab_chunk_size=1024 \
                                beam_clean.decode_only=true \
                                beam_clean.accumulate_gradients=false \
                                beam_clean.learning_rate=0.05 \
                                beam_clean.convergence_tol_pct=0.02 \
                                beam_clean.convergence_patience=5 \
                                beam_clean.memory_threshold=0.8 \
                                evaluate_pii=false &
    sleep 5
done
