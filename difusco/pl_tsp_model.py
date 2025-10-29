"""Lightning module for training the DIFUSCO TSP model."""

import os
import warnings

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
  build_real_adj_from_heatmap,
  decode_tour_from_real_adj,
  deduplicate_tours,
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

    Mode:
      - 'edge' (default): use raw Bernoulli p(edge=1)
      - 'row': normalize per-row so outgoing probs sum to 1 (ignores diag)
    """
    mode = str(getattr(self.args, 'pref_prob_mode', 'edge')).lower()
    eps = 1e-12
    if mode == 'row':
      q = p1_mat.clone()
      n = q.shape[0]
      # Zero diagonal then renormalize rows
      q = q - torch.diag(torch.diag(q))
      row_sum = q.sum(dim=1, keepdim=True).clamp_min(eps)
      q = q / row_sum
      src = q
    else:
      src = p1_mat
    total = 0.0
    for i in range(len(tour) - 1):
      u = int(tour[i])
      v = int(tour[i + 1])
      total = total + torch.log(src[u, v].clamp_min(eps))
    return total

  # ===== Placeholder preference constructor (deterministic + degraded tours) =====
  def _make_worse_tour(self, np_points, tour):
    """Attempt to build a slightly worse tour by applying a degrading mutation.

    Tries a set of 2-opt style segment reversals or simple swaps that increase cost.
    Returns a worse tour (length N+1 with closure) or None if not found.
    """
    tsp = TSPEvaluator(np_points)
    # Work on open sequence without closure
    seq = list(tour[:-1]) if (len(tour) > 1 and tour[0] == tour[-1]) else list(tour)
    n = len(seq)
    if n < 5:
      return None
    base_cost = tsp.evaluate(seq + [seq[0]])

    # Deterministic scan for a pair (i, j) that increases cost via reversal
    # Limit attempts to keep overhead small
    max_checks = min(64, (n * (n - 3)) // 2)
    checks = 0
    for i in range(0, n - 3):
      for j in range(i + 2, n - (0 if i > 0 else 1)):
        if checks >= max_checks:
          break
        # Skip full reversal (i=0, j=n-1) which yields identical tour reversed
        if i == 0 and j == n - 1:
          continue
        new_seq = seq[: i + 1] + list(reversed(seq[i + 1 : j + 1])) + seq[j + 1 :]
        new_cost = tsp.evaluate(new_seq + [new_seq[0]])
        checks += 1
        if new_cost > base_cost + 1e-12:
          return new_seq + [new_seq[0]]
      if checks >= max_checks:
        break

    # Fallback: try a few swaps
    for i in range(1, max(2, n // 8)):
      a = i
      b = n - i - 1
      if a < b and 0 <= a < n and 0 <= b < n:
        new_seq = list(seq)
        new_seq[a], new_seq[b] = new_seq[b], new_seq[a]
        new_cost = tsp.evaluate(new_seq + [new_seq[0]])
        if new_cost > base_cost + 1e-12:
          return new_seq + [new_seq[0]]
    return None

  def _placeholder_preference_pairs(self, np_points, adj_mat_single, pairs_per_graph):
    """Build up to pairs_per_graph preference pairs using a degraded variant of a single tour.

    - Decode deterministic best tour from heatmap
    - Generate up to k worse tours via simple mutations
    Returns list of tuples: [(t_best, t_worse), ...]
    """
    real_adj, _ = build_real_adj_from_heatmap(adj_mat_single[None, ...], np_points,
                                              edge_index_np=None, sparse_graph=False)
    t_best = decode_tour_from_real_adj(real_adj, start_node=0)

    pairs = []
    attempts = 0
    max_attempts = pairs_per_graph * 4
    while len(pairs) < pairs_per_graph and attempts < max_attempts:
      attempts += 1
      t_worse = self._make_worse_tour(np_points, t_best)
      if t_worse is None:
        break
      pairs.append((t_best, t_worse))
    return pairs

  # ===== 2-opt based preference constructor (greedy decode + improvement chain) =====
  def _twoopt_improvement_chain(self, np_points, init_tour, k_steps, device):
    """Generate a chain of tours by applying up to k 2-opt improvements.

    Returns a list [t0, t1, ..., tM] where t0=init_tour, each successive
    tour is obtained by one best 2-opt move, and M <= k_steps. Stops early
    if no improving move is found.
    """
    tsp = TSPEvaluator(np_points)
    chain = [list(init_tour)]
    cur = list(init_tour)
    cur_cost = tsp.evaluate(cur)
    for _ in range(max(0, int(k_steps))):
      batched_tours = np.array([cur], dtype='int64')
      improved, iters = batched_two_opt_torch(
          np_points.astype("float64"), batched_tours, max_iterations=1, device=device)
      nxt = improved[0].tolist()
      nxt_cost = tsp.evaluate(nxt)
      if nxt_cost < cur_cost - 1e-12:
        chain.append(nxt)
        cur, cur_cost = nxt, nxt_cost
      else:
        break
    # Deduplicate in case 2-opt makes no net change in step 0
    chain = deduplicate_tours(chain)
    return chain

  def _twoopt_preference_pairs(self, np_points, adj_mat_single, k_steps, pairs_per_graph, device, pairing='chain'):
    """Build preference pairs from a sequence of 2-opt improvements.

    - Greedy decode a tour from the heatmap
    - Apply up to k 2-opt moves to get multiple tours
    - Form pairs (better, worse) for preference learning

    pairing: 'chain' uses successive pairs (t_{i+1} preferred over t_i).
             'all'   uses all pairs ordered by cost (t_better over t_worse).
    """
    real_adj, _ = build_real_adj_from_heatmap(adj_mat_single[None, ...], np_points,
                                              edge_index_np=None, sparse_graph=False)
    t0 = decode_tour_from_real_adj(real_adj, start_node=0)

    # Build improvement chain
    chain = self._twoopt_improvement_chain(np_points, t0, k_steps=k_steps, device=device)
    if len(chain) <= 1:
      return []

    tsp = TSPEvaluator(np_points)
    tours_with_costs = [(t, tsp.evaluate(t)) for t in chain]

    pairs = []
    if str(pairing).lower() == 'all':
      # Sort by cost ascending; pair every better with every worse below it
      tours_with_costs.sort(key=lambda x: x[1])
      for i in range(len(tours_with_costs)):
        for j in range(i + 1, len(tours_with_costs)):
          if len(pairs) >= pairs_per_graph:
            break
          t_better = tours_with_costs[i][0]
          t_worse = tours_with_costs[j][0]
          pairs.append((t_better, t_worse))
        if len(pairs) >= pairs_per_graph:
          break
    else:
      # Chain mode: only successive improvements
      for i in range(len(chain) - 1):
        if len(pairs) >= pairs_per_graph:
          break
        pairs.append((chain[i + 1], chain[i]))

    return pairs

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

    batch_size = points.shape[0]
    steps = self.args.inference_diffusion_steps
    time_schedule = InferenceSchedule(
        inference_schedule=self.args.inference_schedule,
        T=self.diffusion.T, inference_T=steps)

    # Initialize xt ~ Bernoulli(0.5)
    xt = (torch.randn_like(adj_matrix.float()) > 0).long()

    # Preference application policy (RL)
    apply_last_k_only = bool(getattr(self.args, 'pref_apply_last_k_only', False))
    last_k_steps = int(getattr(self.args, 'pref_last_k_steps', 10))
    if apply_last_k_only:
      used_step_ids = list(range(max(0, steps - last_k_steps), steps))
    else:
      used_step_ids = [steps - 1]

    selected_edge_probs = []  # RL: will hold p(edge=1) for RL-selected steps

    # Mixed precision context if available (reduces VRAM)
    use_mixed = False
    try:
      use_mixed = bool(str(getattr(self.trainer, 'precision', '32')).startswith('16')) or bool(getattr(self.args, 'fp16', False))
    except Exception:
      use_mixed = bool(getattr(self.args, 'fp16', False))

    if not apply_last_k_only:
      # Early steps in pure inference mode (no grads)
      with torch.no_grad():
        for i in range(max(0, steps - 1)):
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
      # Final step with gradients (for RL)
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
      with torch.no_grad():
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        xt = self.categorical_posterior(target_t=t2, t=t1, x0_pred_prob=x0_pred_prob, xt=xt)
    else:
      # General path: compute with grads on RL-selected steps only
      grad_step_ids = set(used_step_ids)
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


    # Decode heatmap and build preferences (2-opt improvements or placeholder)
    pref_loss = torch.tensor(0.0, device=device, requires_grad=True)
    if rl_w > 0.0:
      if self.diffusion_type == 'categorical':
        adj_mat_np = xt.float().cpu().detach().numpy() + 1e-6
      else:
        adj_mat_np = xt.cpu().detach().numpy() * 0.5 + 0.5

      pref_beta = float(getattr(self.args, 'pref_beta', 1.0))
      pairs_per_graph = int(getattr(self.args, 'pref_pairs_per_graph', 1))
      pref_source = str(getattr(self.args, 'pref_source', 'twoopt')).lower()
      pref_2opt_steps = int(getattr(self.args, 'pref_2opt_steps', 4))
      pref_pairing = str(getattr(self.args, 'pref_2opt_pairing', 'chain')).lower()

      batch_pref_losses = []
      mean_margins = []

      total_pairs_batch = 0
      effective_pairs_batch = 0
      for b in range(batch_size):
        np_points = points[b].detach().cpu().numpy()
        if pref_source == 'twoopt':
          pairs = self._twoopt_preference_pairs(
              np_points, adj_mat_np[b], k_steps=pref_2opt_steps,
              pairs_per_graph=pairs_per_graph, device=device, pairing=pref_pairing)
        else:
          # Fallback to degraded tours (previous behavior)
          pairs = self._placeholder_preference_pairs(np_points, adj_mat_np[b], pairs_per_graph)

        total_pairs_count = 0
        effective_pairs_count = 0

        if not pairs:
          warnings.warn(
              f"No placeholder preference pair formed for sample index {b}",
              RuntimeWarning,
          )
          continue

        # Build evaluator for cost-gap checks (if enabled)
        tsp_eval_b = TSPEvaluator(np_points)
        min_gap = float(getattr(self.args, 'pref_min_cost_improve', 0.0))

        for (t_best, t_worse) in pairs:
          total_pairs_count += 1
          # Optional cost-gap gating
          if min_gap > 0.0:
            try:
              gap = float(tsp_eval_b.evaluate(t_worse) - tsp_eval_b.evaluate(t_best))
            except Exception:
              gap = 0.0
            if gap < min_gap:
              continue

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
            mean_margin = torch.stack(margins).mean()
            # Margin gating: only apply if mean_margin <= threshold
            margin_thr = float(getattr(self.args, 'pref_effective_margin', 0.0))
            if mean_margin <= margin_thr:
              effective_pairs_count += 1
              pair_loss = torch.stack(pair_losses).mean()
              batch_pref_losses.append(pair_loss)
              mean_margins.append(mean_margin)

        total_pairs_batch += total_pairs_count
        effective_pairs_batch += effective_pairs_count

      # Log aggregated totals across the batch
      self.log("train/pref_pairs_total", float(total_pairs_batch), on_step=True)
      self.log("train/pref_pairs_effective", float(effective_pairs_batch), on_step=True)
      violate_rate = (effective_pairs_batch / total_pairs_batch) if total_pairs_batch > 0 else 0.0
      self.log("train/pref_violate_rate", float(violate_rate), on_step=True, prog_bar=True)

      # Log selected steps count to audit VRAM invariance in default mode
      self.log("train/pref_selected_steps", float(len(selected_edge_probs)))

      if not batch_pref_losses:
        # Fallback: no pairs produced (unlikely). Keep zero.
        self.log("train/pref_pairs", 0.0, prog_bar=True, on_step=True)
        # Also keep extended counters visible
        self.log("train/pref_pairs_total", 0.0, on_step=True)
        self.log("train/pref_pairs_effective", 0.0, on_step=True)
        self.log("train/pref_violate_rate", 0.0, on_step=True, prog_bar=True)
        self.log("train/pref_margin", 0.0)
        warnings.warn(
            "No preference pairs were formed in this batch (all graphs degenerated to a single unique tour).",
            RuntimeWarning,
        )
      else:
        pref_loss = torch.stack(batch_pref_losses).mean()
        self.log("train/pref_pairs", float(len(batch_pref_losses)), prog_bar=True, on_step=True)
        self.log("train/pref_margin", torch.stack(mean_margins).mean())
    else:
      # No RL preference term requested
      self.log("train/pref_selected_steps", float(len(selected_edge_probs)))
      self.log("train/pref_pairs", 0.0, prog_bar=True, on_step=True)
      self.log("train/pref_pairs_total", 0.0, on_step=True)
      self.log("train/pref_pairs_effective", 0.0, on_step=True)
      self.log("train/pref_violate_rate", 0.0, on_step=True, prog_bar=True)
      self.log("train/pref_margin", 0.0)

    # Log per-step TSP cost from current forward inference heatmap (avg over batch)
    # Uses greedy decode + limited 2-opt (pref_2opt_steps) for light cost.
    try:
      with torch.no_grad():
        adj_mat_np = xt.float().detach().cpu().numpy() + 1e-6 if self.diffusion_type == 'categorical' else (xt.detach().cpu().numpy() * 0.5 + 0.5)
        infer_costs = []
        for b in range(batch_size):
          np_points = points[b].detach().cpu().numpy()
          tours, _ = merge_tours(
              adj_mat_np[b][None, ...], np_points, None,
              sparse_graph=False, parallel_sampling=1,
          )
          solved_tours, _ = batched_two_opt_torch(
              np_points.astype("float64"), np.array(tours).astype('int64'),
              max_iterations=int(min(10, int(getattr(self.args, 'pref_2opt_steps', 4)))), device=device)
          tsp_solver = TSPEvaluator(np_points)
          infer_costs.append(tsp_solver.evaluate(solved_tours[0]))
        if len(infer_costs) > 0:
          self.log("train/infer_cost", float(np.mean(infer_costs)), prog_bar=True, on_step=True, sync_dist=True)
    except Exception as e:
      warnings.warn(f"Failed to compute training-step TSP cost: {e}")

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

    # Note: soft length term removed

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

    # Single deterministic sampling (no sequential/parallel sampling)
    ns, merge_iterations = 0, 0

    xt = torch.randn_like(adj_matrix.float())
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
        parallel_sampling=1,
    )

    # Refine using 2-opt
    solved_tours, ns = batched_two_opt_torch(
        np_points.astype("float64"), np.array(tours).astype('int64'),
        max_iterations=self.args.two_opt_iterations, device=device)

    tsp_solver = TSPEvaluator(np_points)
    gt_cost = tsp_solver.evaluate(np_gt_tour)
    best_solved_cost = tsp_solver.evaluate(solved_tours[0])

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
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')
