#!/bin/bash
 
device=7
export CUDA_VISIBLE_DEVICES=$device
echo "Launching job on CUDA device $device"
    
# Run the python training command in the background.
python beam_clean.py data=papillon data.truncated_seq_len=32 \
                                data.batch_size=2 \
                                run.beam_clean=true \
                                run.nearest_neighbor=true \
                                data.subset_size=50 \
                                privacy.embedding_model_path=gpt2-medium \
                                privacy.const_noise_params.noise_dist=l2_laplace \
                                privacy.const_noise_params.l2_laplace.scale=0.04 \
                                beam_clean.prior_model=gpt2-medium \
                                beam_clean.surrogate_model=l2_laplace \
                                beam_clean.initial_param_val=0.04 \
                                beam_clean.num_epochs=50 \
                                beam_clean.beam_width=20000 \
                                beam_clean.vocab_chunk_size=2048 \
                                beam_clean.decode_only=true \
                                beam_clean.accumulate_gradients=false \
                                beam_clean.learning_rate=0.05 \
                                beam_clean.convergence_tol_pct=0.02 \
                                beam_clean.convergence_patience=5 \
                                beam_clean.memory_threshold=0.8 \
                                evaluate_pii=false \
  