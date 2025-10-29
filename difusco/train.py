"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from utils.progress_bar import KeyedTQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from pl_tsp_model import TSPModel
from pl_mis_model import MISModel


def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  parser.add_argument('--task', type=str, required=True)
  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split (used for MIS).")
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--validation_examples', type=int, default=64)

  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lr_scheduler', type=str, default='constant')

  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='gaussian')
  parser.add_argument('--diffusion_schedule', type=str, default='linear')
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_schedule', type=str, default='linear')
  parser.add_argument('--inference_trick', type=str, default="ddim")
  parser.add_argument('--sequential_sampling', type=int, default=1)
  parser.add_argument('--parallel_sampling', type=int, default=1)

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  # Preference RL fine-tuning (on top of supervised pretraining)
  parser.add_argument('--pref_rl', action='store_true', help='Enable preference RL fine-tuning for TSP.')
  parser.add_argument('--pref_beta', type=float, default=1.0, help='Inverse-temperature for preference loss.')
  parser.add_argument('--pref_pairs_per_graph', type=int, default=1, help='Number of preference pairs per graph (best vs sampled worse).')
  parser.add_argument('--pref_apply_last_k_only', action='store_true', help='Apply preference loss only to the last k denoising steps.')
  parser.add_argument('--pref_last_k_steps', type=int, default=10, help='How many last steps to apply preference loss to.')
  parser.add_argument('--pref_supervised_weight', type=float, default=0.0, help='Optional mixing weight for supervised CE during fine-tune.')
  parser.add_argument('--pref_rl_weight', type=float, default=1.0, help='Weight for preference RL loss when mixed with supervised.')
  parser.add_argument('--pref_effective_margin', type=float, default=0.0,
                      help='Only count/apply pairs whose mean margin <= this threshold as effective.')
  parser.add_argument('--pref_min_cost_improve', type=float, default=0.0,
                      help='Minimum cost improvement (worse_cost - better_cost) to accept a pair.')
  parser.add_argument('--pref_prob_mode', type=str, default='edge',
                      help="Prob mode for tour logprob: 'edge' (raw p(edge=1)) or 'row' (row-normalized).")
  # Preference source and 2-opt generation controls
  parser.add_argument('--pref_source', type=str, default='twoopt',
                      help="Source of preference pairs: 'twoopt' (greedy decode + 2-opt improvements) or 'degrade' (worse mutations).")
  parser.add_argument('--pref_2opt_steps', type=int, default=4,
                      help='Number of incremental 2-opt improvements to attempt when building preference tours.')
  parser.add_argument('--pref_2opt_pairing', type=str, default='chain',
                      help="Pairing strategy for 2-opt tours: 'chain' (successive) or 'all' (all better-worse pairs up to cap).")

  # Note: multi-start decode, tiebreak noise, and training-time sampling removed from TSP.

  # Soft path-length auxiliary loss removed

  # Anchor to pretrained to avoid drift (pure preference fine-tune stabilization)
  parser.add_argument('--pref_anchor_type', type=str, default='none',
                      help="Anchor regularizer type: 'none' or 'l2sp' (weight L2 to pretrained weights).")
  parser.add_argument('--pref_anchor_weight', type=float, default=0.0,
                      help='Weight for anchor regularizer to keep close to pretrained (0 disables).')
  parser.add_argument('--pref_freeze_bottom_layers', type=int, default=0,
                      help='Freeze the first K GNN layers during fine-tuning to reduce drift.')

  parser.add_argument('--project_name', type=str, default='tsp_diffusion')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default=None)
  parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')
  parser.add_argument('--wandb_offline', action='store_true',
                      help='Run Weights & Biases in offline mode (no cloud sync).')

  # Which metrics to display on the TQDM progress bar (comma-separated)
  parser.add_argument('--progress_bar_keys', type=str, default='train/infer_cost,train/pref_pairs,train/pref_violate_rate')

  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_valid_only', action='store_true')

  # Auto-generate TSP instances compatible with dataset format
  parser.add_argument('--auto_generate', action='store_true',
                      help='Automatically generate TSP dataset files before training.')
  parser.add_argument('--auto_num_nodes', type=int, default=50)
  parser.add_argument('--auto_num_train', type=int, default=100000)
  parser.add_argument('--auto_num_val', type=int, default=1000)
  parser.add_argument('--auto_num_test', type=int, default=1000)
  parser.add_argument('--auto_seed', type=int, default=42)
  parser.add_argument('--auto_two_opt_iterations', type=int, default=0,
                      help='Apply 2-opt iterations during generation (0 disables).')
  parser.add_argument('--auto_save_prefix', type=str, default=None,
                      help='Optional file prefix for generated files. Defaults to tsp{N}_*.')
  parser.add_argument('--auto_overwrite', action='store_true')

  args = parser.parse_args()
  return args


def main(args):
  epochs = args.num_epochs
  project_name = args.project_name

  # Optionally auto-generate dataset files
  if args.auto_generate:
    from utils.tsp_instance_gen import generate_split
    tsp_dir = os.path.join(args.storage_path, 'data', 'tsp')
    prefix = args.auto_save_prefix or f"tsp{args.auto_num_nodes}_auto"

    train_path = os.path.join(tsp_dir, f"{prefix}_train.txt")
    val_path = os.path.join(tsp_dir, f"{prefix}_val.txt")
    test_path = os.path.join(tsp_dir, f"{prefix}_test.txt")

    if (not os.path.exists(train_path)) or args.auto_overwrite:
      rank_zero_info(f"Generating train split to {train_path}")
      generate_split(train_path, args.auto_num_train, args.auto_num_nodes, args.auto_seed,
                     two_opt_iterations=args.auto_two_opt_iterations)
    if (not os.path.exists(val_path)) or args.auto_overwrite:
      rank_zero_info(f"Generating val split to {val_path}")
      generate_split(val_path, args.auto_num_val, args.auto_num_nodes, args.auto_seed + 1,
                     two_opt_iterations=args.auto_two_opt_iterations)
    if (not os.path.exists(test_path)) or args.auto_overwrite:
      rank_zero_info(f"Generating test split to {test_path}")
      generate_split(test_path, args.auto_num_test, args.auto_num_nodes, args.auto_seed + 2,
                     two_opt_iterations=args.auto_two_opt_iterations)

    # Redirect training/val/test splits to generated files
    args.training_split = os.path.relpath(train_path, args.storage_path)
    args.validation_split = os.path.relpath(val_path, args.storage_path)
    args.test_split = os.path.relpath(test_path, args.storage_path)

  if args.task == 'tsp':
    model_class = TSPModel
    saving_mode = 'min'
  elif args.task == 'mis':
    model_class = MISModel
    saving_mode = 'max'
  else:
    raise NotImplementedError

  model = model_class(param_args=args)

  # Optionally force W&B offline to avoid cloud sync
  if getattr(args, 'wandb_offline', False):
    os.environ['WANDB_MODE'] = 'offline'

  wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
  wandb_logger = WandbLogger(
      name=args.wandb_logger_name,
      project=project_name,
      entity=args.wandb_entity,
      save_dir=os.path.join(args.storage_path, f'models'),
      id=args.resume_id or wandb_id,
  )
  rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")

  checkpoint_callback = ModelCheckpoint(
      monitor='val/solved_cost', mode=saving_mode,
      save_top_k=3, save_last=True,
      dirpath=os.path.join(wandb_logger.save_dir,
                           args.wandb_logger_name,
                           wandb_logger._id,
                           'checkpoints'),
  )
  lr_callback = LearningRateMonitor(logging_interval='step')

  # Progress bar keys: keep it focused so important metrics show up
  pb_keys = [k.strip() for k in str(getattr(args, 'progress_bar_keys', '')).split(',') if k.strip()]

  trainer = Trainer(
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      max_epochs=epochs,
      callbacks=[KeyedTQDMProgressBar(keys=pb_keys, refresh_rate=20), checkpoint_callback, lr_callback],
      logger=wandb_logger,
      check_val_every_n_epoch=1,
      strategy=DDPStrategy(static_graph=True),
      precision=16 if args.fp16 else 32,
  )

  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path)

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_callback.best_model_path)

  elif args.do_test:
    trainer.validate(model, ckpt_path=ckpt_path)
    if not args.do_valid_only:
      trainer.test(model, ckpt_path=ckpt_path)
  trainer.logger.finalize("success")


if __name__ == '__main__':
  args = arg_parser()
  main(args)
