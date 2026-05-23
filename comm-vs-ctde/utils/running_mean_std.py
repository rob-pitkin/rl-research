import torch


class RunningMeanStd:
    """Welford-style online mean/var over a fixed-shape tensor.

    Adapted from epymarl/components/standarize_stream.py.
    """

    def __init__(self, shape=(), epsilon=1e-4, device="cpu"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self._merge(batch_mean, batch_var, batch_count)

    def _merge(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m_2 / tot
        self.count = tot
