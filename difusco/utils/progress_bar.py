import typing as _t

from pytorch_lightning.callbacks.progress import TQDMProgressBar


class KeyedTQDMProgressBar(TQDMProgressBar):
  """TQDM progress bar that only shows selected metric keys.

  Always keeps 'loss' and 'v_num' if present.
  """

  def __init__(self, keys: _t.Optional[_t.Iterable[str]] = None, refresh_rate: int = 1):
    super().__init__(refresh_rate=refresh_rate)
    self._keys = set(keys or [])

  def get_metrics(self, trainer, pl_module):  # type: ignore[override]
    metrics = super().get_metrics(trainer, pl_module)
    if not self._keys:
      return metrics
    kept = {}
    # Always keep loss and version if available
    for k in ("loss", "v_num"):
      if k in metrics:
        kept[k] = metrics[k]
    for k in self._keys:
      if k in metrics:
        kept[k] = metrics[k]
    return kept

