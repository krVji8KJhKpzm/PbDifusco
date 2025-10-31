export CUDA_VISIBLE_DEVICES=1,2,3

nohup python difusco/train.py \
  --task tsp \
  --storage_path ./temp/ \
  --diffusion_type categorical \
  --sparse_factor -1 \
  --auto_generate \
  --auto_num_nodes 100 \
  --auto_num_train 100000 \
  --auto_num_val 1000 \
  --auto_num_test 1000 \
  --auto_two_opt_iterations 0 \
  --do_train \
  --pref_rl \
  --resume_weight_only \
  --ckpt_path ./checkpoints/tsp100_categorical.ckpt \
  --num_epochs 10 \
  --inference_diffusion_steps 50 \
  --pref_source twoopt \
  --pref_2opt_steps 4 \
  --pref_pairs_per_graph 4 \
  --pref_2opt_pairing all \
  --pref_apply_last_k_only \
  --pref_last_k_steps 4 \
  --pref_rl_weight 0.02 \
  --wandb_logger_name PbDifusco_TSP_Categorical_Pref_RL_2opt \
  --wandb_offline \
  --batch_size 32 \
  --pref_anchor_type l2sp \
  --pref_anchor_weight 5e-3 \
  --pref_freeze_bottom_layers 6 \
  --learning_rate 5e-6 \
  --fp16 \
  --pref_prob_mode edge \
  --pref_effective_margin 0.05 \
  --pref_min_cost_improve 0.01 \
  --progress_bar_keys "train/infer_cost,val/solved_cost,train/pref_pairs,train/pref_pairs_total,train/pref_pairs_effective,train/pref_violate_rate,train/anchor_loss,train/pref_selected_steps" \
  > train_tsp_categorical_pref_rl.out 2>&1 &
