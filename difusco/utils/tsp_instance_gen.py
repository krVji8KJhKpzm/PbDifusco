"""TSP instance generator producing files compatible with TSPGraphDataset.

Each line format:
  x0 y0 x1 y1 ... xN-1 yN-1 output t0 t1 ... tN (1-indexed, closed tour)
"""

import os
import numpy as np
import torch

from .tsp_utils import batched_two_opt_torch


def nearest_neighbor_tour(points: np.ndarray, start: int = None) -> np.ndarray:
  n = points.shape[0]
  if start is None:
    start = int(np.random.randint(0, n))
  unvisited = set(range(n))
  tour = [start]
  unvisited.remove(start)
  while unvisited:
    last = tour[-1]
    # Choose nearest neighbor among unvisited
    dists = np.linalg.norm(points[list(unvisited)] - points[last], axis=1)
    idx = np.argmin(dists)
    nxt = list(unvisited)[idx]
    tour.append(nxt)
    unvisited.remove(nxt)
  tour.append(start)  # close
  return np.array(tour, dtype=np.int64)


def improve_with_two_opt(points: np.ndarray, tour: np.ndarray, iterations: int = 0) -> np.ndarray:
  if iterations <= 0:
    return tour
  # batched_two_opt_torch expects batch dimension
  improved, _ = batched_two_opt_torch(points.astype("float64"), tour[None, :].astype("int64"),
                                      max_iterations=iterations, device="cpu")
  return improved[0]


def write_dataset_file(path: str, samples):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w') as f:
    for pts, tour in samples:
      coords = ' '.join(['{:.6f} {:.6f}'.format(x, y) for x, y in pts])
      # Dataset expects 1-indexed
      tour_1 = (tour + 1).tolist()
      tour_str = ' '.join(str(t) for t in tour_1)
      f.write(f"{coords} output {tour_str}\n")


def generate_split(path: str, num_samples: int, n_nodes: int, seed: int,
                   two_opt_iterations: int = 0):
  rng = np.random.default_rng(seed)
  samples = []
  for _ in range(num_samples):
    pts = rng.random((n_nodes, 2), dtype=np.float64)
    tour = nearest_neighbor_tour(pts)
    tour = improve_with_two_opt(pts, tour, iterations=two_opt_iterations)
    samples.append((pts, tour))
  write_dataset_file(path, samples)
  return path
