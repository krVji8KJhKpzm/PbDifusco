"""Lightning module for training the DIFUSCO TSP model."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import (
  TSPEvaluator,
  batched_two_opt_torch,
  merge_tours,
  multi_start_tours,
  build_real_adj_from_heatmap,
  decode_tour_from_real_adj,
)


class TSPModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    self.train_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.test_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
        sparse_factor=self.args.sparse_factor,
    )

    self.validation_dataset = TSPGraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        sparse_factor=self.args.sparse_factor,
    )

    # Pretrained anchor snapshot (built lazily on fit start)
    self._anchor_params = None

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)

  def on_fit_start(self):
    # Optionally freeze bottom K GNN layers
    k = int(getattr(self.args, 'pref_freeze_bottom_layers', 0))
    if k > 0:
      layers = getattr(self.model, 'layers', None)
      if layers is not None:
        for idx in range(min(k, len(layers))):
          for p in layers[idx].parameters():
            p.requires_grad = False
        rank_zero_info(f"Froze bottom {min(k, len(layers))} GNN layers for fine-tuning")

    # Build anchor snapshot if requested
    anchor_type = str(getattr(self.args, 'pref_anchor_type', 'none')).lower()
    anchor_w = float(getattr(self.args, 'pref_anchor_weight', 0.0))
    if anchor_type == 'l2sp' and anchor_w > 0.0 and self._anchor_params is None:
      self._anchor_params = {
          n: p.detach().clone().to(p.device) for n, p in self.model.named_parameters() if p.requires_grad
      }
      rank_zero_info(f"Built L2SP anchor snapshot for {len(self._anchor_params)} params.")

  def _anchor_loss(self):
    anchor_type = str(getattr(self.args, 'pref_anchor_type', 'none')).lower()
    anchor_w = float(getattr(self.args, 'pref_anchor_weight', 0.0))
    if anchor_type == 'l2sp' and anchor_w > 0.0 and self._anchor_params is not None:
      loss = 0.0
      for n, p in self.model.named_parameters():
        if p.requires_grad and n in self._anchor_params:
          ref = self._anchor_params[n]
          # Ensure device match
          if ref.device != p.device:
            ref = ref.to(p.device)
            self._anchor_params[n] = ref
          loss = loss + torch.sum((p - ref) ** 2)
      return anchor_w * loss
    return None

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    if not self.sparse:
      _, points, adj_matrix, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
    else:
      _, graph_data, point_indicator, edge_indicator, _ = batch
      t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))

    # Sample from diffusion
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    if self.sparse:
      t = torch.from_numpy(t).float()
      t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
      xt = xt.reshape(-1)
      adj_matrix = adj_matrix.reshape(-1)
      points = points.reshape(-1, 2)
      edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)
    else:
      t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    # Compute loss
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(x0_pred, adj_matrix.long())
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    # Preference RL fine-tune path (on top of supervised pretraining)
    if getattr(self.args, 'pref_rl', False):
      return self.preference_training_step(batch, batch_idx)
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def _edge_probs_from_logits(self, x0_pred, points):
    """Convert logits to Bernoulli p(edge=1) matrix per sample (dense graphs)."""
    if self.sparse:
      raise ValueError("Preference RL currently supports dense TSP only (sparse not supported).")
    x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
    p1 = x0_pred_prob[..., 1]
    return p1

  def _tour_logprob_from_edge_probs(self, p1_mat, tour):
    """Sum log-prob over directed edges along a closed tour.
    p1_mat: (N, N) torch tensor of p(edge=1)
    tour: list/array of node indices, length N+1 with closure
    """
    eps = 1e-9
    total = 0.0
    for i in range(len(tour) - 1):
      u = int(tour[i])
      v = int(tour[i + 1])
      total = total + torch.log(p1_mat[u, v].clamp_min(eps))
    return total

  def preference_training_step(self, batch, batch_idx):
    if self.diffusion_type != 'categorical':
      raise ValueError("Preference RL fine-tuning is implemented for categorical diffusion only.")
    if self.sparse:
      # Could be extended; keep scope tight per request (TSP dense)
      raise ValueError("Preference RL currently supports dense TSP only (set sparse_factor<=0)")

    # Unpack batch (dense case)
    real_batch_idx, points, adj_matrix, gt_tour = batch
    device = points.device

    # Weights/config (now from self.args)
    sup_w = float(getattr(self.args, 'pref_supervised_weight', 0.0))
    rl_w = float(getattr(self.args, 'pref_rl_weight', 1.0))
    softlen_weight = float(getattr(self.args, 'pref_softlen_weight', 0.0))
    softlen_degree_lambda = float(getattr(self.args, 'pref_softlen_degree_lambda', 0.1))

    batch_size = points.shape[0]
    steps = self.args.inference_diffusion_steps
    time_schedule = InferenceSchedule(
        inference_schedule=self.args.inference_schedule,
        T=self.diffusion.T, inference_T=steps)

    # Initialize xt ~ Bernoulli(0.5)
    xt = (torch.randn_like(adj_matrix.float()) > 0).long()

    # Preference application policy (RL) and Soft-length selection
    apply_last_k_only = bool(getattr(self.args, 'pref_apply_last_k_only', False))
    last_k_steps = int(getattr(self.args, 'pref_last_k_steps', 10))
    if apply_last_k_only:
      used_step_ids = list(range(max(0, steps - last_k_steps), steps))
    else:
      used_step_ids = [steps - 1]

    soft_apply_last_k_only = bool(getattr(self.args, 'pref_softlen_apply_last_k_only', False))
    soft_last_k_steps = int(getattr(self.args, 'pref_softlen_last_k_steps', 10))
    if soft_apply_last_k_only:
      soft_used_step_ids = list(range(max(0, steps - soft_last_k_steps), steps))
    else:
      soft_used_step_ids = [steps - 1]

    selected_edge_probs = []  # RL: will hold p(edge=1) for RL-selected steps
    softlen_edge_probs = []   # Softlen: p(edge=1) for soft-selected steps

    # Mixed precision context if available (reduces VRAM)
    use_mixed = False
    try:
      use_mixed = bool(str(getattr(self.trainer, 'precision', '32')).startswith('16')) or bool(getattr(self.args, 'fp16', False))
    except Exception:
      use_mixed = bool(getattr(self.args, 'fp16', False))

    if not apply_last_k_only:
      # Fast path with optional softlen last-k gradients
      last_k_start = max(0, steps - soft_last_k_steps) if soft_apply_last_k_only else steps - 1
      # 1) Early steps in pure inference mode (no grads)
      with torch.no_grad():
        for i in range(max(0, last_k_start)):
          t1, t2 = time_schedule(i)
          t1 = np.array([t1]).astype(int)
          t2 = np.array([t2]).astype(int)
          t_tensor = torch.from_numpy(t1).view(1)
          with torch.cuda.amp.autocast(enabled=use_mixed):
            x0_pred = self.forward(
                points.float().to(device),
                xt.float().to(device),
                t_tensor.float().to(device),
                None,
            )
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
          xt = self.categorical_posterior(target_t=t2, t=t1, x0_pred_prob=x0_pred_prob, xt=xt)
      # 2) Optional last-k-1 steps with gradients for softlen
      for i in range(max(0, last_k_start), max(0, steps - 1)):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)
        t_tensor = torch.from_numpy(t1).view(1)
        with torch.cuda.amp.autocast(enabled=use_mixed):
          # xt may be created under inference_mode earlier; clone to get a normal tensor for autograd
          xt_in = xt.clone()
          x0_pred = self.forward(
              points.float().to(device),
              xt_in.float().to(device),
              t_tensor.float().to(device),
              None,
          )
        if i in soft_used_step_ids:
          p1_soft = self._edge_probs_from_logits(x0_pred, points)
          softlen_edge_probs.append((i, p1_soft))
        with torch.no_grad():
          x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
          xt = self.categorical_posterior(target_t=t2, t=t1, x0_pred_prob=x0_pred_prob, xt=xt)
      # 3) Final step with gradients (RL always uses at least this)
      i = steps - 1
      t1, t2 = time_schedule(i)
      t1 = np.array([t1]).astype(int)
      t2 = np.array([t2]).astype(int)
      t_tensor = torch.from_numpy(t1).view(1)
      with torch.cuda.amp.autocast(enabled=use_mixed):
        xt_in = xt.clone()
        x0_pred = self.forward(
            points.float().to(device),
            xt_in.float().to(device),
            t_tensor.float().to(device),
            None,
        )
      p1_final = self._edge_probs_from_logits(x0_pred, points)
      selected_edge_probs.append((i, p1_final))
      if i in soft_used_step_ids:
        softlen_edge_probs.append((i, p1_final))
      with torch.no_grad():
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        xt = self.categorical_posterior(target_t=t2, t=t1, x0_pred_prob=x0_pred_prob, xt=xt)
    else:
      # General path: compute with grads on union(RL last-k, Soft last-k)
      grad_step_ids = set(used_step_ids).union(set(soft_used_step_ids))
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)
        if i in grad_step_ids:
          t_tensor = torch.from_numpy(t1).view(1)
          with torch.cuda.amp.autocast(enabled=use_mixed):
            xt_in = xt.clone()
            x0_pred = self.forward(
                points.float().to(device),
                xt_in.float().to(device),
                t_tensor.float().to(device),
                None,
            )
          # Record RL p1
          if i in used_step_ids:
            p1_rl = self._edge_probs_from_logits(x0_pred, points)
            selected_edge_probs.append((i, p1_rl))
          # Record softlen p1
          if i in soft_used_step_ids:
            p1_soft = self._edge_probs_from_logits(x0_pred, points)
            softlen_edge_probs.append((i, p1_soft))
          with torch.no_grad():
            x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            xt = self.categorical_posterior(target_t=t2, t=t1, x0_pred_prob=x0_pred_prob, xt=xt)
        else:
          with torch.no_grad():
            t_tensor = torch.from_numpy(t1).view(1)
            with torch.cuda.amp.autocast(enabled=use_mixed):
              x0_pred = self.forward(
                  points.float().to(device),
                  xt.float().to(device),
                  t_tensor.float().to(device),
                  None,
              )
              x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            xt = self.categorical_posterior(target_t=t2, t=t1, x0_pred_prob=x0_pred_prob, xt=xt)

    # Soft path-length auxiliary loss from heatmap, averaged over selected steps
    softlen_loss = None
    softlen_length_mean = None
    softlen_degree_mean = None
    if softlen_edge_probs:
      # Precompute pairwise distances (detach to avoid grads through points)
      dist_mat = torch.cdist(points.float(), points.float(), p=2).detach()
      B, N, _ = dist_mat.shape
      diag_mask = (1.0 - torch.eye(N, device=device)).unsqueeze(0)

      acc_losses = []
      acc_len = []
      acc_deg = []
      for (_, p1_batched) in softlen_edge_probs:
        p = p1_batched.to(dist_mat.dtype)
        p_sym = 0.5 * (p + p.transpose(-1, -2))
        p_sym = p_sym * diag_mask  # zero diagonal

        # Length: sum over undirected edges once
        length_b = 0.5 * (p_sym * dist_mat).sum(dim=(-2, -1))  # (B,)
        # Degree regularization
        deg_b = p_sym.sum(dim=-1)  # (B, N)
        deg_term_b = ((deg_b - 2.0) ** 2).mean(dim=-1)  # (B,)

        acc_len.append(length_b.mean())
        acc_deg.append(deg_term_b.mean())
        acc_losses.append(length_b.mean() + softlen_degree_lambda * deg_term_b.mean())

      softlen_length_mean = torch.stack(acc_len).mean()
      softlen_degree_mean = torch.stack(acc_deg).mean()
      softlen_loss = torch.stack(acc_losses).mean()

    # Log soft loss components irrespective of weight for visibility
    if softlen_loss is not None:
      self.log("train/softlen_length", softlen_length_mean)
      self.log("train/softlen_degree", softlen_degree_mean)
      self.log("train/softlen_loss", softlen_loss)
      self.log("train/softlen_weight", float(softlen_weight))
      self.log("train/softlen_selected_steps", float(len(softlen_edge_probs)))

    # Decode heatmap and build preferences from multi-start greedy paths
    pref_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if rl_w > 0.0:
      if self.diffusion_type == 'categorical':
        adj_mat_np = xt.float().cpu().detach().numpy() + 1e-6
      else:
        adj_mat_np = xt.cpu().detach().numpy() * 0.5 + 0.5

      # Train-only decode mode: sampling vs tie-break (default)
      use_sampling_decode = bool(getattr(self.args, 'train_decode_sampling', False))
      use_last_k_for_sampling = bool(getattr(self.args, 'train_sampling_use_last_k', False))
      K_sampling = max(1, int(getattr(self.args, 'train_sampling_K', 1)))

      pref_beta = float(getattr(self.args, 'pref_beta', 1.0))
      num_starts_cfg = int(getattr(self.args, 'pref_num_start_nodes', 8))
      pairs_per_graph = int(getattr(self.args, 'pref_pairs_per_graph', 1))

      batch_pref_losses = []
      mean_margins = []

      for b in range(batch_size):
        np_points = points[b].detach().cpu().numpy()
        n = np_points.shape[0]
        # Sample start nodes (unique)
        num_starts = min(num_starts_cfg, n)
        start_nodes = torch.randperm(n)[:num_starts].tolist()
        if use_sampling_decode:
          # Use predicted edge probabilities to sample heatmaps,
          # then decode greedily (deterministic) from multiple start nodes.
          tours = []

          # Build a dict {step_id: p1_batched} for steps where we recorded probabilities
          p1_by_step = {int(si): p1b for (si, p1b) in selected_edge_probs} if selected_edge_probs else {}

          # Decide which steps to use for sampling
          if use_last_k_for_sampling and len(p1_by_step) > 0:
            # Intersect recorded steps with RL selected steps to be safe
            step_ids = [s for s in used_step_ids if s in p1_by_step]
            if not step_ids and len(p1_by_step) > 0:
              # Fallback to whatever we have (e.g., only final step was recorded)
              step_ids = [sorted(p1_by_step.keys())[-1]]
          else:
            # Only use final step (if available); fallback to empty list
            if len(p1_by_step) > 0:
              step_ids = [sorted(p1_by_step.keys())[-1]]
            else:
              step_ids = []

          rng = np.random.default_rng()

          if not step_ids:
            # No probabilities recorded (unexpected). Fallback: use xt heatmap once.
            sampled_adj = adj_mat_np[b]
            real_adj, _ = build_real_adj_from_heatmap(sampled_adj[None, ...], np_points,
                                                      edge_index_np=None, sparse_graph=False,
                                                      random_tiebreak=False)
            for s in start_nodes:
              t = decode_tour_from_real_adj(real_adj, start_node=s, random_tiebreak=False)
              tours.append(t)
          else:
            # For each chosen step, sample K heatmaps and decode from multiple starts
            for si in step_ids:
              p1_batched = p1_by_step[si]            # (B, N, N)
              p1_np = p1_batched[b].detach().float().cpu().numpy()
              p1_np = np.clip(p1_np, 0.0, 1.0)
              for _k in range(K_sampling):
                sampled_adj = (rng.random(p1_np.shape) < p1_np).astype(np.float32)
                real_adj, _ = build_real_adj_from_heatmap(sampled_adj[None, ...], np_points,
                                                          edge_index_np=None, sparse_graph=False,
                                                          random_tiebreak=False)
                for s in start_nodes:
                  t = decode_tour_from_real_adj(real_adj, start_node=s, random_tiebreak=False)
                  tours.append(t)
        else:
          # Original path: multi-start greedy with optional stochastic tie-break
          tours, _ = multi_start_tours(
              adj_mat_np[b:b+1], np_points, edge_index_np=None,
              start_nodes=start_nodes, sparse_graph=False,
              random_tiebreak=bool(getattr(self.args, 'pref_decode_random_tiebreak', False)),
              noise_scale=float(getattr(self.args, 'pref_decode_noise_scale', 1e-3)))

        # Evaluate costs
        tsp_solver = TSPEvaluator(np_points)
        costs = [tsp_solver.evaluate(t) for t in tours]
        best_idx = int(np.argmin(costs))
        worse_pool = [i for i in range(len(tours)) if i != best_idx]
        if not worse_pool:
          continue  # degenerate case

        # Form preference pairs: (best, sampled worse)
        worse_sel = worse_pool if len(worse_pool) <= pairs_per_graph else list(np.random.choice(worse_pool, pairs_per_graph, replace=False))

        for wi in worse_sel:
          t_best = tours[best_idx]
          t_worse = tours[wi]

          # Aggregate loss across selected steps
          pair_losses = []
          margins = []
          for (_, p1_batched) in selected_edge_probs:
            p1_mat = p1_batched[b]  # (N, N)
            logp_best = self._tour_logprob_from_edge_probs(p1_mat, t_best)
            logp_worse = self._tour_logprob_from_edge_probs(p1_mat, t_worse)
            margin = logp_best - logp_worse
            margins.append(margin.detach())
            # Pairwise preference: -log(sigmoid(beta * (logp_best - logp_worse)))
            pair_losses.append(F.softplus(-pref_beta * margin))

          if pair_losses:
            pair_loss = torch.stack(pair_losses).mean()
            batch_pref_losses.append(pair_loss)
            mean_margins.append(torch.stack(margins).mean())

      # Log selected steps count to audit VRAM invariance in default mode
      self.log("train/pref_selected_steps", float(len(selected_edge_probs)))

      if not batch_pref_losses:
        # Fallback: no pairs produced (unlikely). Keep zero.
        self.log("train/pref_pairs", 0.0)
        self.log("train/pref_margin", 0.0)
      else:
        pref_loss = torch.stack(batch_pref_losses).mean()
        self.log("train/pref_pairs", float(len(batch_pref_losses)))
        self.log("train/pref_margin", torch.stack(mean_margins).mean())
    else:
      # No RL preference term requested
      self.log("train/pref_selected_steps", float(len(selected_edge_probs)))
      self.log("train/pref_pairs", 0.0)
      self.log("train/pref_margin", 0.0)

    # Optional supervised CE mixing during fine-tune
    total_loss = None
    if sup_w > 0.0:
      ce_loss = self.categorical_training_step(batch, batch_idx)
      total_loss = sup_w * ce_loss
    else:
      total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Add RL preference term
    if rl_w > 0.0:
      self.log("train/pref_loss", pref_loss)
      total_loss = total_loss + rl_w * pref_loss

    # Add soft length term
    if softlen_weight > 0.0 and softlen_loss is not None:
      total_loss = total_loss + softlen_weight * softlen_loss

    # Anchor regularization to pretrained weights to avoid drift
    anchor_reg = self._anchor_loss()
    if anchor_reg is not None:
      total_loss = total_loss + anchor_reg
      self.log("train/anchor_loss", anchor_reg)
      self.log("train/anchor_weight", float(getattr(self.args, 'pref_anchor_weight', 0.0)))

    self.log("train/total_loss", total_loss)
    return total_loss

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    stacked_tours = []
    ns, merge_iterations = 0, 0

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)
        else:
          xt = self.categorical_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)

      if self.diffusion_type == 'gaussian':
        adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
      else:
        adj_mat = xt.float().cpu().detach().numpy() + 1e-6

      if self.args.save_numpy_heatmap:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      tours, merge_iterations = merge_tours(
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )

      # Refine using 2-opt
      solved_tours, ns = batched_two_opt_torch(
          np_points.astype("float64"), np.array(tours).astype('int64'),
          max_iterations=self.args.two_opt_iterations, device=device)
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost = np.min(all_solved_costs)

    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": ns,
        f"{split}/merge_iterations": merge_iterations,
    }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
