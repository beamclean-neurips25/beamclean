#!/bin/bash
 
device=5
export CUDA_VISIBLE_DEVICES=$device
echo "Launching job on CUDA device $device"
    
# Run the python training command in the background.
python beam_clean.py data=papillon data.truncated_seq_len=64 \
                                data.batch_size=2 \
                                run.beam_clean=true \
                                run.nearest_neighbor=true \
                                data.subset_size=50 \
                                privacy.const_noise_params.gauss.std=0.5 \
                                privacy.const_noise_params.gauss.load_from_file=false \
                                beam_clean.surrogate_model=gaussian \
                                beam_clean.initial_param_val=0.5 \
                                beam_clean.num_epochs=10 \
                                beam_clean.beam_width=10 \
                                beam_clean.vocab_chunk_size=1024 \
                                beam_clean.decode_only=true \
                                beam_clean.accumulate_gradients=false \
                                beam_clean.learning_rate=0.05 \
                                evaluate_pii=false
