"""Automated two-phase training for TSP: supervised pretrain + preference RL fine-tune.

This wrapper composes two invocations of train.py while preserving the same
CLI interface as much as possible.
"""

import argparse
import os
import sys
import uuid
import subprocess


def build_base_args(parser: argparse.ArgumentParser):
  # Mirror the core args from difusco/train.py to keep interface consistent
  parser.add_argument('--task', type=str, default='tsp')
  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None)
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--validation_examples', type=int, default=64)

  parser.add_argument('--batch_size', type=int, default=64)
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
  parser.add_argument('--inference_trick', type=str, default='ddim')
  parser.add_argument('--sequential_sampling', type=int, default=1)
  parser.add_argument('--parallel_sampling', type=int, default=1)

  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  parser.add_argument('--project_name', type=str, default='tsp_diffusion')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default=None)
  parser.add_argument('--resume_id', type=str, default=None)
  parser.add_argument('--wandb_offline', action='store_true')
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  # Preference RL args to forward into fine-tune stage
  parser.add_argument('--pref_beta', type=float, default=1.0)
  parser.add_argument('--pref_num_start_nodes', type=int, default=8)
  parser.add_argument('--pref_pairs_per_graph', type=int, default=1)
  parser.add_argument('--pref_apply_last_k_only', action='store_true')
  parser.add_argument('--pref_last_k_steps', type=int, default=10)
  parser.add_argument('--pref_supervised_weight', type=float, default=0.0)
  parser.add_argument('--pref_rl_weight', type=float, default=1.0)
  parser.add_argument('--pref_decode_random_tiebreak', action='store_true')
  parser.add_argument('--pref_decode_noise_scale', type=float, default=1e-3)

  # Auto-train orchestration knobs
  parser.add_argument('--auto_supervised_epochs', type=int, default=50)
  parser.add_argument('--auto_pref_epochs', type=int, default=10)
  parser.add_argument('--auto_do_test', action='store_true')


def to_argv(args: argparse.Namespace, phase_epochs: int, extra_flags: list):
  argv = [
      '--task', args.task,
      '--storage_path', args.storage_path,
      '--training_split', args.training_split,
      '--validation_split', args.validation_split,
      '--test_split', args.test_split,
      '--validation_examples', str(args.validation_examples),

      '--batch_size', str(args.batch_size),
      '--num_epochs', str(phase_epochs),
      '--learning_rate', str(args.learning_rate),
      '--weight_decay', str(args.weight_decay),
      '--lr_scheduler', args.lr_scheduler,

      '--diffusion_type', args.diffusion_type,
      '--diffusion_schedule', args.diffusion_schedule,
      '--diffusion_steps', str(args.diffusion_steps),
      '--inference_diffusion_steps', str(args.inference_diffusion_steps),
      '--inference_schedule', args.inference_schedule,
      '--inference_trick', args.inference_trick,
      '--sequential_sampling', str(args.sequential_sampling),
      '--parallel_sampling', str(args.parallel_sampling),

      '--n_layers', str(args.n_layers),
      '--hidden_dim', str(args.hidden_dim),
      '--sparse_factor', str(args.sparse_factor),
      '--aggregation', args.aggregation,
      '--two_opt_iterations', str(args.two_opt_iterations),

      '--project_name', args.project_name,
      '--wandb_logger_name', args.wandb_logger_name or 'auto_train_pref',
      '--resume_id', args.resume_id,
  ]
  if args.wandb_offline:
    argv += ['--wandb_offline']

  if args.training_split_label_dir is not None:
    argv += ['--training_split_label_dir', args.training_split_label_dir]
  if args.wandb_entity is not None:
    argv += ['--wandb_entity', args.wandb_entity]
  if args.save_numpy_heatmap:
    argv += ['--save_numpy_heatmap']
  if args.fp16:
    argv += ['--fp16']
  if args.use_activation_checkpoint:
    argv += ['--use_activation_checkpoint']

  # Preference args (only meaningful in fine-tune stage, but safe to pass always)
  argv += [
      '--pref_beta', str(args.pref_beta),
      '--pref_num_start_nodes', str(args.pref_num_start_nodes),
      '--pref_pairs_per_graph', str(args.pref_pairs_per_graph),
      '--pref_last_k_steps', str(args.pref_last_k_steps),
      '--pref_supervised_weight', str(args.pref_supervised_weight),
      '--pref_rl_weight', str(args.pref_rl_weight),
  ]
  if args.pref_apply_last_k_only:
    argv += ['--pref_apply_last_k_only']
  if args.pref_decode_random_tiebreak:
    argv += ['--pref_decode_random_tiebreak']
  argv += ['--pref_decode_noise_scale', str(args.pref_decode_noise_scale)]

  argv += extra_flags
  return argv


def run_train(invocation_argv):
  train_py = os.path.join(os.path.dirname(__file__), 'train.py')
  cmd = [sys.executable, train_py] + invocation_argv
  print('Launching:', ' '.join(cmd))
  subprocess.run(cmd, check=True)


def main():
  parser = argparse.ArgumentParser(description='Auto supervised+pref-RL training')
  build_base_args(parser)
  args = parser.parse_args()

  if args.task != 'tsp':
    raise ValueError('This auto-trainer currently supports only TSP.')

  # Stable run id for both phases unless provided
  if not args.resume_id:
    args.resume_id = f"auto-{uuid.uuid4().hex[:8]}"

  # Phase 1: supervised pretrain
  pretrain_flags = ['--do_train']
  if args.auto_do_test:
    pretrain_flags.append('--do_test')
  run_train(to_argv(args, args.auto_supervised_epochs, pretrain_flags))

  # Compute checkpoint path for fine-tune (last.ckpt)
  ckpt_path = os.path.join(
      args.storage_path, 'models',
      (args.wandb_logger_name or 'auto_train_pref'),
      args.resume_id, 'checkpoints', 'last.ckpt')

  # Phase 2: preference RL fine-tune (resume weights only)
  finetune_flags = ['--do_train', '--pref_rl', '--resume_weight_only', '--ckpt_path', ckpt_path]
  if args.auto_do_test:
    finetune_flags.append('--do_test')
  run_train(to_argv(args, args.auto_pref_epochs, finetune_flags))


if __name__ == '__main__':
  main()
