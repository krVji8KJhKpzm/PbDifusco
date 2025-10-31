export CUDA_VISIBLE_DEVICES=1

python -u difusco/train.py --task "tsp" --wandb_logger_name "tsp_diffusion_graph_categorical_tsp100_test" --diffusion_type "categorical" --do_test --learning_rate 0.0002 --weight_decay 0.0001 --lr_scheduler "cosine-decay" --storage_path "./temp" --training_split "./data/tsp/tsp100_auto_train.txt" --validation_split "./data/tsp/tsp100_auto_val.txt" --test_split "./data/tsp/tsp100_auto_test.txt" --batch_size 32 --num_epochs 25 --inference_schedule "cosine" --inference_diffusion_steps 50 --ckpt_path "./temp/models/epoch=1-2opt.ckpt" --resume_weight_only --wandb_offline

# 7.79761 TSP100 2epoch

# python -u difusco/train.py --task "tsp" --wandb_logger_name "tsp_diffusion_graph_categorical_tsp100_test" --diffusion_type "categorical" --do_test --learning_rate 0.0002 --weight_decay 0.0001 --lr_scheduler "cosine-decay" --storage_path "./temp" --training_split "./data/tsp/tsp100_auto_train.txt" --validation_split "./data/tsp/tsp100_auto_val.txt" --test_split "./data/tsp/tsp100_auto_test.txt" --batch_size 32 --num_epochs 25 --inference_schedule "cosine" --inference_diffusion_steps 50 --ckpt_path "./checkpoints/tsp100_categorical.ckpt" --resume_weight_only --wandb_offline

# 7.79921 TSP100 baseline