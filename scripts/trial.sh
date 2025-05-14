#!/bin/bash

python beam_clean.py data=papillon data.truncated_seq_len=128 \
                            data.batch_size=2 \
                            run.beam_clean=true \
                            run.nearest_neighbor=true \
                            data.subset_size=100 \
                            privacy.embedding_model_path="../LLama-3.2-1B-Instruct" \
                            privacy.const_noise_params.noise_dist=l1_laplace \
                            privacy.const_noise_params.l1_laplace.scale=0.296 \
                            beam_clean.candidate_token_ids_path=null \
                            beam_clean.prior_model="../LLama-3.2-1B-Instruct" \
                            beam_clean.surrogate_model=l1_laplace \
                            beam_clean.initial_param_val=0.296 \
                            beam_clean.num_epochs=50 \
                            beam_clean.beam_width=100 \
                            beam_clean.vocab_chunk_size=2048 \
                            beam_clean.decode_only=true \
                            beam_clean.accumulate_gradients=false \
                            beam_clean.learning_rate=0.05 \
                            beam_clean.convergence_tol_pct=0.02 \
                            beam_clean.convergence_patience=5 \
                            beam_clean.memory_threshold=0.8 \
                            evaluate_pii=true \