import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
from utils.cython_merge.cython_merge import merge_cython


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  return tour, iterator


def numpy_merge(points, adj_mat):
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)
  merge_iterations = 0
  for edge in (-adj_mat / dists).flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
    real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
        ).toarray() for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          cython_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations


class TSPEvaluator(object):
  def __init__(self, points):
    self.dist_mat = scipy.spatial.distance_matrix(points, points)

  def evaluate(self, route):
    total_cost = 0
    for i in range(len(route) - 1):
      total_cost += self.dist_mat[route[i], route[i + 1]]
    return total_cost


# ===== New helpers for multi-start greedy decoding =====
def numpy_merge_stochastic(points, adj_mat, noise_scale=1e-3, rng=None):
  """Greedy edge insertion with stochastic tie-break by adding noise to ranking."""
  if rng is None:
    rng = np.random.default_rng()
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)

  # Scores: prefer large adj and small distance
  scores = -adj_mat / (dists + 1e-9)
  jitter = rng.normal(loc=0.0, scale=noise_scale, size=scores.shape)
  scores = scores + jitter

  merge_iterations = 0
  for edge in scores.flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


def build_real_adj_from_heatmap(adj_mat, np_points, edge_index_np=None, sparse_graph=False,
                                random_tiebreak=False, noise_scale=1e-3):
  """Build the symmetric real adjacency matrix via greedy merge from a heatmap.

  Returns (real_adj_mat, merge_iterations).
  """
  if not sparse_graph:
    sym_adj = adj_mat[0] + adj_mat[0].T
  else:
    if edge_index_np is None:
      raise ValueError("edge_index_np is required for sparse graphs")
    sym_adj = scipy.sparse.coo_matrix(
        (adj_mat, (edge_index_np[0], edge_index_np[1])),
    ).toarray() + scipy.sparse.coo_matrix(
        (adj_mat, (edge_index_np[1], edge_index_np[0])),
    ).toarray()

  # Prefer stochastic numpy merge when randomness is requested; else use Cython
  if random_tiebreak:
    real_adj_mat, merge_iterations = numpy_merge_stochastic(np_points, sym_adj, noise_scale=noise_scale)
  else:
    real_adj_mat, merge_iterations = cython_merge(np_points, sym_adj)
  return real_adj_mat, merge_iterations


def decode_tour_from_real_adj(real_adj_mat, start_node=0, random_tiebreak=False, rng=None):
  """Decode a single tour from a 0/1 adjacency matrix starting at `start_node`.

  If `random_tiebreak` is True, when two neighbors are available (first step),
  pick randomly instead of deterministically.
  """
  n = real_adj_mat.shape[0]
  start_node = int(start_node % n)
  tour = [start_node]
  if rng is None:
    rng = np.random.default_rng()
  while len(tour) < n + 1:
    neighbors = np.nonzero(real_adj_mat[tour[-1]])[0]
    if len(tour) > 1:
      neighbors = neighbors[neighbors != tour[-2]]
    if random_tiebreak and len(neighbors) > 1 and len(tour) == 1:
      # Randomly choose direction on the cycle only at the first step
      idx = rng.integers(low=0, high=len(neighbors))
      next_node = neighbors[idx]
    else:
      # Deterministic tie-breaker: pick max index (matches original)
      next_node = neighbors.max()
    tour.append(int(next_node))
  return tour


def multi_start_tours(adj_mat, np_points, edge_index_np=None, start_nodes=None, sparse_graph=False,
                      random_tiebreak=False, noise_scale=1e-3):
  """Generate tours by greedy decoding from multiple start nodes on the same heatmap.

  Args:
    adj_mat: (1, N, N) heatmap if dense or (E,) if sparse
    np_points: (N, 2)
    edge_index_np: (2, E) for sparse graphs
    start_nodes: iterable of distinct start node indices
    sparse_graph: whether input is sparse

  Returns:
    tours: list of tours in the same index space as np_points
    merge_iterations: float
  """
  real_adj_mat, merge_iterations = build_real_adj_from_heatmap(
      adj_mat, np_points, edge_index_np=edge_index_np, sparse_graph=sparse_graph,
      random_tiebreak=random_tiebreak, noise_scale=noise_scale)

  n = np_points.shape[0]
  if start_nodes is None:
    start_nodes = [0]
  # Sanitize and uniquify
  start_nodes = sorted({int(s % n) for s in start_nodes})
  rng = np.random.default_rng()
  tours = [decode_tour_from_real_adj(real_adj_mat, s, random_tiebreak=random_tiebreak, rng=rng)
           for s in start_nodes]
  return tours, merge_iterations


# ===== Tour canonicalization and de-duplication =====
def canonicalize_tour(tour):
  """Return a canonical tuple representation of a TSP tour (cycle).

  The canonicalization ignores the closing node (if repeated), is invariant to
  rotation and reversal, and returns the lexicographically smallest sequence.
  """
  # Convert to list of ints
  if isinstance(tour, np.ndarray):
    seq = tour.tolist()
  else:
    seq = list(tour)

  # Drop closing node if present (e.g., [0, 1, 2, 0])
  if len(seq) >= 2 and seq[0] == seq[-1]:
    seq = seq[:-1]

  n = len(seq)
  if n == 0:
    return tuple()

  # Generate rotations for both directions
  def rotations(s):
    for r in range(n):
      yield s[r:] + s[:r]

  forward = seq
  backward = list(reversed(seq))
  candidates = list(rotations(forward)) + list(rotations(backward))
  # Choose lexicographically smallest
  best = min(tuple(c) for c in candidates)
  return best


def deduplicate_tours(tours):
  """Deduplicate tours by canonical form, preserving first occurrence order.

  Args:
    tours: list of tours (each a list/array of node indices; may include closing node)

  Returns:
    unique_tours: list of tours with duplicates (up to rotation/reversal) removed
  """
  seen = set()
  unique_tours = []
  for t in tours:
    key = canonicalize_tour(t)
    if key in seen:
      continue
    seen.add(key)
    unique_tours.append(t)
  return unique_tours
